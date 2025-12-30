# Mistake Log (Do Not Erase)

## 2025-12-30 — Green confidence became non-discriminative because it was computed *after* top-p/top-k truncation

### Symptom
- Even with `green_conf_thresh = 0`, a large number of tokens were marked green very early (e.g., `green=128` at step 2).
- The threshold appeared to have “no discrimination power”.

### Root Cause
We computed the “raw confidence” used for green/red gating **on a distribution that had already been truncated by top-p/top-k**.

In particular, with `top_p=0.9`, if the top-1 token already has probability ≥ 0.9, nucleus filtering keeps **only 1 token**, making the distribution effectively one-hot.

For `ALG=entropy` (where we used negative entropy as “confidence”):

- If the distribution is one-hot: entropy = 0
- Negative entropy = 0
- Therefore, `raw_confidence >= 0` becomes true for many positions

Result: the confidence gate collapses, and stability alone can produce many greens.

### Why this was subtle
- We thought we were using “raw (pre-temperature) confidence”, but we were still applying top-p/top-k filtering before computing that confidence.
- The truncation changes the distribution so drastically that it invalidates the intended semantics of the threshold.

### Fix (Correct Approach)
For green/red decisions, compute confidence from the **full, unfiltered, pre-temperature logits**.

A robust choice that makes thresholds meaningful:
- Use the **log-probability of the chosen token** under the full distribution:

  - `raw_confidence = log_softmax(logits_raw)[x0]` (log-prob)

This keeps thresholds stable and interpretable:
- `0` is essentially impossible (only if probability = 1)
- Values are negative; e.g. `-3` corresponds to about `p ≈ e^{-3} ≈ 0.05`

### Extra Practical Note
- Store `cand_conf_full` in `float32` (not `bfloat16`) to avoid score quantization/rounding reducing discrimination.

### Checklist (to avoid repeating this)
- When a threshold “does nothing”, confirm:
  - Are we computing the score on a *filtered* distribution (top_p/top_k)?
  - Is the score type consistent with the threshold scale (prob vs log-prob vs neg-entropy)?
  - Is dtype/precision collapsing values (bf16/fp16)?
- If using entropy-based measures, beware of **truncation causing entropy→0**.

### Minimal Debug Recipe
- In the run log, verify the exact arguments actually passed (`alg`, `top_p`, `green_conf_thresh`, etc.).
- Compare distributions before/after truncation for a few tokens and confirm whether top_p is collapsing to a single candidate.
