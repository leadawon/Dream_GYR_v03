# Dream_GYR_v02 (Dream-based green/red decoding)

This repository is a **Dream (diffusion LM) based variant** that adds a **green/red skeleton decoding policy**.

At each diffusion step, masked positions are classified into:
- **green**: positions whose candidate token is considered sufficiently stable/confident
- **red**: the remaining masked positions (fallback pool)

The decoder then commits tokens by selecting from green first, and filling any remaining quota from red.

## What’s Different From Baseline Dream

### Green/Red policy (skeleton decoding)
- Green membership is computed per-step using token stability/oscillation trackers and a confidence threshold.
- Stability is re-checked **every step** (no “once green, always green” caching).
- Token *selection* within the green/red pools follows the baseline confidence-based selection behavior.

### Forward-pass cost tracking (`fp_stats.json`)
- Optional tracking of forward-pass cost per generation call.
- Records `configured_steps` vs `executed_steps` (actual forwards can be smaller with early-stop).

### Early stop (policy-gated)
- If there are no `[MASK]` tokens left, decoding can stop early.
- To preserve baseline safety, early-stop can be gated to only trigger when green/red policy is enabled.

### Debug logging
- Optional per-step log line: `step=... mask=... green=... red=...`
- Scripts tee stdout/stderr into a mirrored `logs/` tree.

## Quick Start: GSM8K (lm_eval)

Run the provided scripts under `eval_instruct/`:

```
cd /workspace/Dream_GYR_v02/eval_instruct

# policy0 (baseline path)
./run_gpu2_gyr_green_red_gsm8k.sh

# policy1 (green/red policy)
./run_gpu3_gyr_green_red_gsm8k.sh
```

## Outputs

Each run writes to:
- `out/gyr_green_red/gsm8k/policy{0|1}/limit{N}/alg{...}_seed{...}_ts{...}_steps{...}_newtok{...}_bs{...}/`
- `fp_stats.json` lives inside that run folder.
- Logs are mirrored under the same structure rooted at `logs/` (e.g., `logs/.../run.log`).

## Notes

- If you want identical behavior to the baseline Dream decoding, disable the policy (`enable_green_red_policy=0`).
- The green/red policy is intended for research/debugging of decoding schedules and cost/quality tradeoffs.
