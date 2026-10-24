---
inclusion: always
---

# Scientific Integrity Principles for DIVERGE

These rules apply to all work in this repo: scoring logic, UI copy, experiments, exports, and docs. If a request conflicts with them, say so and propose a version that complies.

## 1. Divergence does not equal intelligence
- Distance from AI baselines (`semantic_divergence`, `information_gain`, claim novelty) measures how different an answer is, not how good it is.
- Originality must stay gated by quality (see `GATED` and the gate in `diverge/scoring.py`). Never remove or weaken the gate to let off-topic, incoherent, or random text score higher.
- Keep the deliberately bad demo seeds in `diverge/seed.py` (`demo:noise`, `demo:offtopic`, `demo:turing`) as sanity checks. A change that makes them score well is a regression.

## 2. AI consensus is not ground truth
- Baselines in `data/baselines.json` come from a small local model (Qwen2.5-0.5B). Treat them as a reference point for novelty only.
- Never use baselines or any LLM output to decide correctness. Correctness needs human-authored references with sources (e.g. the planned `data/references.json`).
- Don't word UI text as if agreeing or disagreeing with the AI makes an answer right or wrong.

## 3. Evaluate novelty independently from correctness
- Compute and report originality and quality/correctness as separate signals, then combine them explicitly. Don't merge them into one opaque number.
- A correct conventional answer must not be scored as worthless, and a fluent wrong contrarian answer must not score as high value.
- When an answer can't be verified, label it `unverified`. Don't guess.

## 4. Reward evidence-based reasoning, not disagreement alone
- Reasoning quality and evidence quality (mechanisms, quantities, named examples, sources, testable predictions) are what earn credit for divergence.
- Heuristics like regex evidence patterns and NLI contradiction are proxies. Call them heuristics in code comments and UI, and don't present them as validated measurements.
- Scoring weights are experimental (`diverge-score-0.1`, "NOT validated"). Any weight or formula change must bump `VERSION` and be documented in the module docstring.

## 5. Estimated vs. measured learning value
- Per-submission values (information gain, marginal gain proxies, Intelligence Value) are estimates. Label them that way.
- Only results produced by actually running `diverge/experiment.py` and stored in `data/experiment/results.json` count as measured.
- Keep the MEASURED (green) vs. HYPOTHESIS (amber) distinction in the UI (`dv-measured` / `dv-hypo`). State scope limits honestly: small proxy model, CommonsenseQA, 5 seeds, no significance at p < 0.05, not trained on DIVERGE submissions.

## 6. Never fabricate energy savings or model performance
- Never hard-code, invent, extrapolate, or "illustrate" accuracy, FLOPs, energy, kWh, CO₂, or cost numbers as if they were results.
- Every number shown as a result must be computed from stored data or a run that can be reproduced, with its source traceable in code.
- The climate link (data efficiency → less compute → less energy) is a hypothesis. Present it as one unless an experiment in this repo measures it.
- If a result is negative or null (e.g. divergent selection does not beat random), report it plainly. Don't omit or reframe it.
- `engine/data.py` and `engine/interventions.py` hold SIMULATED placeholders. Keep the label, and keep that data clearly separate from real data (`engine/realdata.py`).

## 7. Preserve existing working functionality
- Extend, don't replace. Existing tabs in `app.py`, the store schema in `data/submissions.jsonl`, export formats in `diverge/store.py`, and cached baselines must keep working.
- `engine/` and `unbound/` (the earlier UNBOUND app) stay untouched unless the user explicitly asks.
- New evaluation fields must be additive and backward compatible: old records without them must still load, render, and export.
- Everything runs locally with no API keys and no user data leaving the machine. Keep it that way by default.
- After changes, check that `streamlit run app.py` still loads and that `.venv/bin/python -m diverge.seed` still scores without errors. Target Python 3.9 (keep `from __future__ import annotations`).
