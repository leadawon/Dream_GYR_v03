# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import copy
import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)


def _safe_int_env(name: str, default: int = 0) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _record_fp_stats(
    *,
    stats_path: str,
    alg: str,
    steps: int,
    batch_size: int,
    max_length: int,
    forward_passes_per_example: int,
    green_red_policy: Dict[str, Any],
    append: bool,
    configured_steps: Optional[int] = None,
    executed_steps: Optional[int] = None,
    early_stopped: Optional[bool] = None,
    early_stop_step: Optional[int] = None,
):
    """Aggregate forward-pass statistics into a single JSON file.

    Notes:
    - We treat the cost as "forward passes per example" (not per-batch), to keep
      the metric comparable across batch sizes.
    - This is intentionally lightweight: minimum information needed for
      quality-speed tradeoff tracking.
    """
    target_dir = os.path.dirname(stats_path) or "."
    os.makedirs(target_dir, exist_ok=True)

    if append and os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    runs = data.get("runs")
    if not isinstance(runs, dict):
        runs = {}

    # Optional append-only log for later analysis/debugging.
    entries = data.get("entries")
    if not isinstance(entries, list):
        entries = []

    gr_enabled = bool(green_red_policy.get("enable_green_red_policy", False))
    config_key = f"alg={alg}_steps={int(steps)}_gr={int(gr_enabled)}"

    run = runs.get(config_key)
    if not isinstance(run, dict):
        run = {
            "config": {
                "alg": alg,
                "steps": int(steps),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "green_red_policy": dict(green_red_policy),
            },
            "total_examples": 0,
            "total_forward_passes": 0,
            "total_dependency_forward_passes": 0,
            "min_forward_passes": None,
            "max_forward_passes": None,
        }

    n_examples = max(int(batch_size), 0)
    cfg_steps = int(configured_steps) if configured_steps is not None else int(steps)
    exec_steps = int(executed_steps) if executed_steps is not None else int(forward_passes_per_example)
    fp = int(forward_passes_per_example)

    # Keep forward-pass accounting consistent with executed steps.
    fp = exec_steps

    run["total_examples"] += n_examples
    run["total_forward_passes"] += fp * n_examples
    run["total_dependency_forward_passes"] += 0

    if run["min_forward_passes"] is None or fp < run["min_forward_passes"]:
        run["min_forward_passes"] = fp
    if run["max_forward_passes"] is None or fp > run["max_forward_passes"]:
        run["max_forward_passes"] = fp

    total_examples = max(int(run["total_examples"]), 1)
    run["avg_forward_passes"] = run["total_forward_passes"] / total_examples
    run["avg_dependency_forward_passes"] = 0.0
    run["avg_total_estimated"] = run["avg_forward_passes"] + run["avg_dependency_forward_passes"]

    # Backwards-safe metadata (latest observed values).
    run["configured_steps"] = cfg_steps
    run["executed_steps"] = exec_steps
    run["last_configured_steps"] = cfg_steps
    run["last_executed_steps"] = exec_steps
    if early_stopped is not None:
        run["last_early_stopped"] = bool(early_stopped)
        run["early_stopped"] = bool(early_stopped)
    if early_stop_step is not None:
        run["last_early_stop_step"] = int(early_stop_step)
        run["early_stop_step"] = int(early_stop_step)

    runs[config_key] = run
    data["runs"] = runs
    now = datetime.now().isoformat()
    data["last_update"] = now

    if append:
        entries.append(
            {
                "timestamp": now,
                "alg": alg,
                "steps": int(steps),
                "configured_steps": int(cfg_steps),
                "executed_steps": int(exec_steps),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "forward_passes_total": int(exec_steps),
                "avg_forward_passes_per_example": int(exec_steps),
                "early_stopped": bool(early_stopped) if early_stopped is not None else (exec_steps < cfg_steps),
                "early_stop_step": int(early_stop_step) if early_stop_step is not None else (-1),
                "green_red_policy": dict(green_red_policy),
            }
        )
        data["entries"] = entries

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
    return_raw_confidence=False,
):
    logits_raw = logits

    logits_temp = logits
    if temperature > 0:
        logits_temp = logits_temp / temperature

    # Apply nucleus / top-k filtering consistently for sampling, and also for raw confidence
    if top_p is not None and top_p < 1:
        logits_temp = top_p_logits(logits_temp, top_p)
    if top_k is not None:
        logits_temp = top_k_logits(logits_temp, top_k)
    probs = torch.softmax(logits_temp, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    if not return_raw_confidence:
        return confidence, x0

    # Raw confidence for green/red decisions: log-probability of the chosen token
    # under the *full* (unfiltered), pre-temperature distribution.
    # This keeps thresholds meaningful (e.g., -3) and avoids top_p/top_k truncation
    # collapsing scores to 0 for many positions.
    log_probs_raw_full = F.log_softmax(logits_raw.float(), dim=-1)
    raw_confidence = torch.gather(log_probs_raw_full, -1, x0.unsqueeze(-1)).squeeze(-1)
    return confidence, x0, raw_confidence


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    generation_stats: Optional[Dict[str, Any]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # Optional debug logging for green/red policy
        self.enable_gr_step_logging: bool = bool(kwargs.pop("enable_gr_step_logging", False))
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # green/red skeleton decoding policy (default OFF)
        self.enable_green_red_policy: bool = bool(kwargs.pop("enable_green_red_policy", False))
        # green/red policy params
        # NOTE: for entropy alg (neg-entropy), confidence is <= 0. Threshold -3 means only fairly confident positions.
        self.green_conf_thresh: float = float(kwargs.pop("green_conf_thresh", -3.0))
        self.green_min_stable_steps: int = int(kwargs.pop("green_min_stable_steps", 2))
        self.green_max_osc: int = int(kwargs.pop("green_max_osc", 0))
        self.red_min_stable_steps: int = int(kwargs.pop("red_min_stable_steps", 3))
        self.greenest_force_unmask: bool = bool(kwargs.pop("greenest_force_unmask", True))
        self.greenest_score_mode: str = str(kwargs.pop("greenest_score_mode", "confidence"))

        # yellow candidate selection policy (default OFF)
        # NOTE: raw_confidence is log-prob; closer to 0 means more confident.
        # "Looser" threshold means a more negative value.
        self.enable_yellow_policy: bool = bool(kwargs.pop("enable_yellow_policy", False))
        self.yellow_conf_thresh: float = float(kwargs.pop("yellow_conf_thresh", -4.5))
        self.yellow_min_remaining_masks: int = int(kwargs.pop("yellow_min_remaining_masks", 32))
        self.yellow_frac_of_remaining: float = float(kwargs.pop("yellow_frac_of_remaining", 0.15))
        # If <= 0, treat as unlimited (no fixed upper cap).
        self.yellow_max_per_row: int = int(kwargs.pop("yellow_max_per_row", -1))
        # If k_row_raw < this cap, select 0 (skip testing) for that row.
        self.yellow_min_selected_cap: int = int(kwargs.pop("yellow_min_selected_cap", 8))
        self.yellow_min_stable_steps: int = int(kwargs.pop("yellow_min_stable_steps", 1))

        # early stop when all masks are resolved (default ON, but gated to GR policy by default)
        self.enable_early_stop_when_no_mask: bool = bool(kwargs.pop("enable_early_stop_when_no_mask", True))
        self.early_stop_only_when_gr_enabled: bool = bool(kwargs.pop("early_stop_only_when_gr_enabled", True))

        # forward-pass stats (quality-speed tracking)
        self.enable_fp_stats: bool = bool(kwargs.pop("enable_fp_stats", False))
        self.fp_stats_path: Optional[str] = kwargs.pop("fp_stats_path", None)
        self.fp_stats_append: bool = bool(kwargs.pop("fp_stats_append", True))

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        gr_enabled = bool(getattr(generation_config, "enable_green_red_policy", False))
        enable_yellow_policy = bool(getattr(generation_config, "enable_yellow_policy", False))
        enable_fp_stats = bool(getattr(generation_config, "enable_fp_stats", False))
        fp_stats_path = getattr(generation_config, "fp_stats_path", None)
        fp_stats_append = bool(getattr(generation_config, "fp_stats_append", True))

        green_conf_thresh = float(getattr(generation_config, "green_conf_thresh", -1.0))
        green_min_stable_steps = int(getattr(generation_config, "green_min_stable_steps", 2))
        green_max_osc = int(getattr(generation_config, "green_max_osc", 0))
        red_min_stable_steps = int(getattr(generation_config, "red_min_stable_steps", 3))
        greenest_force_unmask = bool(getattr(generation_config, "greenest_force_unmask", True))
        greenest_score_mode = str(getattr(generation_config, "greenest_score_mode", "confidence"))
        enable_gr_step_logging = bool(getattr(generation_config, "enable_gr_step_logging", False))

        # Yellow candidate selection (subset of red_pool). This phase does NOT commit/unmask.
        yellow_conf_thresh = float(getattr(generation_config, "yellow_conf_thresh", -4.5))
        yellow_min_remaining_masks = int(getattr(generation_config, "yellow_min_remaining_masks", 32))
        yellow_frac_of_remaining = float(getattr(generation_config, "yellow_frac_of_remaining", 0.15))
        yellow_max_per_row = int(getattr(generation_config, "yellow_max_per_row", -1))
        yellow_min_selected_cap = int(getattr(generation_config, "yellow_min_selected_cap", 8))
        yellow_min_stable_steps = int(getattr(generation_config, "yellow_min_stable_steps", 1))

        # Yellow is defined as a subset of the red pool; it only makes sense when GR policy is enabled.
        yellow_enabled = bool(enable_yellow_policy and gr_enabled)

        enable_early_stop_when_no_mask = bool(getattr(generation_config, "enable_early_stop_when_no_mask", True))
        early_stop_only_when_gr_enabled = bool(getattr(generation_config, "early_stop_only_when_gr_enabled", True))

        generation_stats: Optional[Dict[str, Any]] = None
        if return_dict_in_generate:
            generation_stats = {
                "gr_enabled": bool(gr_enabled),
                "yellow_enabled": bool(yellow_enabled),
                "enable_early_stop_when_no_mask": bool(enable_early_stop_when_no_mask),
                "early_stop_only_when_gr_enabled": bool(early_stop_only_when_gr_enabled),
                "green_conf_thresh": green_conf_thresh,
                "green_min_stable_steps": green_min_stable_steps,
                "green_max_osc": green_max_osc,
                "red_min_stable_steps": red_min_stable_steps,
                "greenest_force_unmask": bool(greenest_force_unmask),
                "greenest_score_mode": greenest_score_mode,
                "yellow_conf_thresh": float(yellow_conf_thresh) if yellow_enabled else None,
                "yellow_min_remaining_masks": int(yellow_min_remaining_masks) if yellow_enabled else None,
                "yellow_frac_of_remaining": float(yellow_frac_of_remaining) if yellow_enabled else None,
                "yellow_max_per_row": int(yellow_max_per_row) if yellow_enabled else None,
                "yellow_min_selected_cap": int(yellow_min_selected_cap) if yellow_enabled else None,
                "yellow_min_stable_steps": int(yellow_min_stable_steps) if yellow_enabled else None,
                "configured_steps": int(steps),
                "executed_steps": 0,
                "early_stopped": False,
                "early_stop_step": -1,
                "green_count_per_step": [],
                "red_commit_ok_count_per_step": [],
                "step_green_count": [],
                "step_committed_count": [],
                "step_fallback_used": [],
                "step_yellow_candidate_count": [] if yellow_enabled else None,
                "step_yellow_selected_count": [] if yellow_enabled else None,
                "forced_greenest_used_count_per_step": [],
                "forced_greenest_used_count_total": 0,
            }

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # green/red policy token-state trackers (no extra forward passes)
        if gr_enabled:
            bsz, seqlen = x.shape
            prev_cand_token = torch.full((bsz, seqlen), -1, device=x.device, dtype=torch.long)
            stable_run_len = torch.zeros((bsz, seqlen), device=x.device, dtype=torch.int32)
            osc_count = torch.zeros((bsz, seqlen), device=x.device, dtype=torch.int32)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)

        executed_steps = 0
        early_stopped = False
        early_stop_step = -1
        for i in range(steps):
            mask_index = (x == mask_token_id)

            # Early stop BEFORE forward: if there are no [MASK] tokens left, further steps cannot change x.
            if enable_early_stop_when_no_mask:
                if early_stop_only_when_gr_enabled:
                    if gr_enabled and (not torch.any(mask_index)):
                        early_stopped = True
                        early_stop_step = int(i)
                        break
                else:
                    if not torch.any(mask_index):
                        early_stopped = True
                        early_stop_step = int(i)
                        break

            logits = self(x, attention_mask, tok_idx).logits
            executed_steps += 1
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            forced_before = None
            if generation_stats is not None and gr_enabled:
                forced_before = int(generation_stats.get("forced_greenest_used_count_total", 0))

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0, raw_confidence = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        return_raw_confidence=True,
                    )
                elif alg == 'topk_margin':
                    confidence, x0, raw_confidence = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        margin_confidence=True,
                        return_raw_confidence=True,
                    )
                elif alg == 'entropy':
                    confidence, x0, raw_confidence = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        neg_entropy=True,
                        return_raw_confidence=True,
                    )
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)

                # IMPORTANT: when green/red policy is OFF, keep the original logic unchanged.
                if not gr_enabled:
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        x[row_indices,transfer_index] = x_[row_indices,transfer_index]
                else:
                    bsz, seqlen = x.shape

                    # Stall protection: if the baseline schedule would transfer 0 tokens due to truncation,
                    # force progress by allowing at least 1 token transfer (policy-on only).
                    if number_transfer_tokens <= 0 and greenest_force_unmask and torch.any(mask_index):
                        number_transfer_tokens = 1

                    cand_token_full = torch.full((bsz, seqlen), -1, device=x.device, dtype=torch.long)
                    cand_conf_full = torch.full((bsz, seqlen), -torch.inf, device=x.device, dtype=torch.float32)
                    cand_token_full[mask_index] = x0
                    # Use RAW (pre-temperature) confidence for green/red decisions.
                    cand_conf_full[mask_index] = raw_confidence

                    # IMPORTANT: token *selection* (which positions to unmask) should follow the baseline logic.
                    # So we use the baseline `confidence` values (computed after temperature/top_p/top_k and with
                    # the selected token semantics) for ranking/sampling positions.
                    sel_conf_full = torch.full((bsz, seqlen), -torch.inf, device=x.device, dtype=torch.float32)
                    sel_conf_full[mask_index] = confidence.float()

                    # Update stability/oscillation trackers on masked positions
                    same = (cand_token_full == prev_cand_token) & mask_index
                    diff = (cand_token_full != prev_cand_token) & mask_index & (prev_cand_token != -1)
                    stable_run_len[same] += 1
                    stable_run_len[mask_index & ~same] = 1
                    osc_count[diff] += 1
                    prev_cand_token[mask_index] = cand_token_full[mask_index]

                    is_green = (
                        mask_index
                        & (stable_run_len >= green_min_stable_steps)
                        & (cand_conf_full >= green_conf_thresh)
                        & (osc_count <= green_max_osc)
                    )

                    # Red = the remainder after strict green filtering.
                    # This matches the requested behavior: within a step, pick green strictly first,
                    # then fill the rest of the transfer quota from all non-green masked positions.
                    is_red_commit_ok = mask_index & (~is_green)

                    forced_rows_cnt = 0

                    # Yellow selection outputs (no commit). Keep these as plain Python ints when disabled
                    # to avoid any extra CUDA work or side effects in the OFF path.
                    step_yellow_cand_cnt = 0
                    step_yellow_sel_cnt = 0

                    # New behavior (v03 goal): commit ALL green positions in this step (no quota / no top-k).
                    # Keep policy-off path unchanged above.
                    if i == steps - 1:
                        # Final step: commit all remaining masks to guarantee termination.
                        commit_mask = mask_index
                    else:
                        commit_mask = is_green.clone()

                        # Fallback: if a row has zero green positions, commit exactly one from red/masked.
                        # This preserves progress without changing the main behavior.
                        row_green = commit_mask.sum(dim=-1)
                        row_mask = mask_index.sum(dim=-1)
                        empty_rows = (row_mask > 0) & (row_green == 0)

                        if torch.any(empty_rows):
                            if greenest_score_mode != "confidence":
                                # Only confidence mode is implemented for now.
                                pass

                            # Pick the best masked position by baseline confidence.
                            row_best_vals, row_best_idx = sel_conf_full.max(dim=-1)
                            row_has_pick = torch.isfinite(row_best_vals) & empty_rows
                            if torch.any(row_has_pick):
                                commit_mask[row_has_pick, row_best_idx[row_has_pick]] = True
                                forced_rows_cnt = int(row_has_pick.sum().item())
                                if generation_stats is not None:
                                    generation_stats["forced_greenest_used_count_total"] += forced_rows_cnt

                        # Yellow candidate selection (subset of remaining mask after commit_mask).
                        # This block must not change x/commit decisions.
                        if yellow_enabled:
                            remain_mask = mask_index & (~commit_mask)
                            yellow_candidate = (
                                remain_mask
                                & (cand_conf_full >= yellow_conf_thresh)
                                & (stable_run_len >= yellow_min_stable_steps)
                            )

                            # Select up to k_row per row, where k_row increases with remaining masks.
                            remain_cnt_per_row = remain_mask.sum(dim=-1)
                            yellow_selected_mask = torch.zeros_like(remain_mask)

                            for r in range(int(bsz)):
                                remain_cnt = int(remain_cnt_per_row[r].item())
                                if remain_cnt < int(yellow_min_remaining_masks):
                                    continue

                                k_row_raw = int(yellow_frac_of_remaining * float(remain_cnt))
                                # If too small, skip entirely (0 selected) to avoid expensive/noisy tests.
                                if k_row_raw < int(yellow_min_selected_cap):
                                    continue

                                # Optional upper cap: only applied when > 0. If <= 0, treat as unlimited.
                                if yellow_max_per_row is not None and int(yellow_max_per_row) > 0:
                                    k_row = min(int(k_row_raw), int(yellow_max_per_row))
                                else:
                                    k_row = int(k_row_raw)

                                cand_pos = torch.nonzero(yellow_candidate[r], as_tuple=False).squeeze(-1)
                                if cand_pos.numel() == 0:
                                    continue

                                # Pick top-k by raw confidence among candidates.
                                cand_scores = cand_conf_full[r, cand_pos]
                                k_eff = min(int(k_row), int(cand_pos.numel()))
                                _, top_local = torch.topk(cand_scores, k_eff)
                                top_pos = cand_pos[top_local]
                                yellow_selected_mask[r, top_pos] = True

                            step_yellow_cand_cnt = int(yellow_candidate.sum().item())
                            step_yellow_sel_cnt = int(yellow_selected_mask.sum().item())

                    # Apply committed updates.
                    if torch.any(commit_mask):
                        x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        x[commit_mask] = x_[commit_mask]

                    if generation_stats is not None:
                        generation_stats["green_count_per_step"].append(int(is_green.sum().item()))
                        generation_stats["red_commit_ok_count_per_step"].append(int(is_red_commit_ok.sum().item()))
                        step_green_count = int(is_green.sum().item())
                        step_committed_count = int(commit_mask.sum().item())
                        step_fallback_used = int((step_green_count == 0) and (step_committed_count == 1))
                        generation_stats["step_green_count"].append(step_green_count)
                        generation_stats["step_committed_count"].append(step_committed_count)
                        generation_stats["step_fallback_used"].append(step_fallback_used)

                        if yellow_enabled and generation_stats.get("step_yellow_candidate_count") is not None:
                            generation_stats["step_yellow_candidate_count"].append(int(step_yellow_cand_cnt))
                        if yellow_enabled and generation_stats.get("step_yellow_selected_count") is not None:
                            generation_stats["step_yellow_selected_count"].append(int(step_yellow_sel_cnt))

                        forced_after = int(generation_stats.get("forced_greenest_used_count_total", 0))
                        forced_delta = forced_after - (forced_before or 0)
                        generation_stats["forced_greenest_used_count_per_step"].append(int(forced_delta))
                        max_keep = 64
                        for kname in [
                            "green_count_per_step",
                            "red_commit_ok_count_per_step",
                            "step_green_count",
                            "step_committed_count",
                            "step_fallback_used",
                            "forced_greenest_used_count_per_step",
                        ]:
                            if len(generation_stats[kname]) > max_keep:
                                generation_stats[kname] = generation_stats[kname][-max_keep:]

                        if yellow_enabled:
                            for kname in ["step_yellow_candidate_count", "step_yellow_selected_count"]:
                                if generation_stats.get(kname) is not None and len(generation_stats[kname]) > max_keep:
                                    generation_stats[kname] = generation_stats[kname][-max_keep:]

                    if enable_gr_step_logging:
                        try:
                            mask_cnt = int(mask_index.sum().item())
                            green_cnt = int(is_green.sum().item())
                            red_ok_cnt = int(is_red_commit_ok.sum().item())
                            if yellow_enabled:
                                print(
                                    f"step={i+1}/{steps} mask={mask_cnt} green={green_cnt} red={red_ok_cnt} "
                                    f"yellow_cand={int(step_yellow_cand_cnt)} yellow_sel={int(step_yellow_sel_cnt)}",
                                    flush=True,
                                )
                            else:
                                # Preserve the exact log line format when yellow is disabled.
                                print(f"step={i+1}/{steps} mask={mask_cnt} green={green_cnt} red={red_ok_cnt}", flush=True)
                        except Exception:
                            pass

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())

        if generation_stats is not None:
            generation_stats["executed_steps"] = int(executed_steps)
            generation_stats["early_stopped"] = bool(early_stopped or (executed_steps < int(steps)))
            generation_stats["early_stop_step"] = int(early_stop_step if early_stopped else (-1))
        
        if return_dict_in_generate:
            # Record forward-pass stats once per generation call (rank0 only to avoid contention)
            if enable_fp_stats and fp_stats_path and _safe_int_env("RANK", _safe_int_env("LOCAL_RANK", 0)) == 0:
                try:
                    _record_fp_stats(
                        stats_path=str(fp_stats_path),
                        alg=str(alg),
                        steps=int(steps),
                        batch_size=int(x.shape[0]),
                        max_length=int(x.shape[1]),
                        forward_passes_per_example=int(executed_steps),
                        green_red_policy={
                            "enable_green_red_policy": bool(gr_enabled),
                            "green_conf_thresh": green_conf_thresh,
                            "green_min_stable_steps": green_min_stable_steps,
                            "green_max_osc": green_max_osc,
                            "red_min_stable_steps": red_min_stable_steps,
                            "greenest_force_unmask": bool(greenest_force_unmask),
                            "greenest_score_mode": greenest_score_mode,
                        },
                        append=bool(fp_stats_append),
                        configured_steps=int(steps),
                        executed_steps=int(executed_steps),
                        early_stopped=bool(early_stopped or (executed_steps < int(steps))),
                        early_stop_step=int(early_stop_step if early_stopped else (-1)),
                    )
                except Exception as e:
                    logger.warning(f"Failed to record fp_stats to {fp_stats_path}: {e}")

            return DreamModelOutput(
                sequences=x,
                history=histories,
                generation_stats=generation_stats,
            )
        else:
            # Even when not returning a dict, we may still want fp_stats.
            if enable_fp_stats and fp_stats_path and _safe_int_env("RANK", _safe_int_env("LOCAL_RANK", 0)) == 0:
                try:
                    _record_fp_stats(
                        stats_path=str(fp_stats_path),
                        alg=str(alg),
                        steps=int(steps),
                        batch_size=int(x.shape[0]),
                        max_length=int(x.shape[1]),
                        forward_passes_per_example=int(executed_steps),
                        green_red_policy={
                            "enable_green_red_policy": bool(gr_enabled),
                            "green_conf_thresh": green_conf_thresh,
                            "green_min_stable_steps": green_min_stable_steps,
                            "green_max_osc": green_max_osc,
                            "red_min_stable_steps": red_min_stable_steps,
                            "greenest_force_unmask": bool(greenest_force_unmask),
                            "greenest_score_mode": greenest_score_mode,
                        },
                        append=bool(fp_stats_append),
                        configured_steps=int(steps),
                        executed_steps=int(executed_steps),
                        early_stopped=bool(early_stopped or (executed_steps < int(steps))),
                        early_stop_step=int(early_stop_step if early_stopped else (-1)),
                    )
                except Exception as e:
                    logger.warning(f"Failed to record fp_stats to {fp_stats_path}: {e}")
            return x