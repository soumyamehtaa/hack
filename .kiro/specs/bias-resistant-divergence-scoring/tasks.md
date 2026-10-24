# Implementation Plan: Bias-Resistant Divergence Scoring

## Overview

Python 3.9 (`from __future__ import annotations`). The backend comes first in small additive steps, then the four-page `app.py` redesign. Every task leaves the app runnable on the existing Streamlit server on :8501, which hot-reloads. Never start a second server. `engine/` and `unbound/` stay untouched. `data/submissions.jsonl` is never rewritten, so record its sha256 before starting and compare it at each checkpoint (`shasum -a 256 data/submissions.jsonl`). `diverge/value.py` (`ivs-0.1-experimental`) already exists and is reused unchanged.

## Tasks

- [ ] 1. Backend: originality-only D and evaluation config
  - [ ] 1.1 Add Divergence_Score_D to `diverge/scoring.py`
    - Add a pure helper `divergence_d(div, cnov, chal) -> float | None`: 100 × (0.50·div + 0.30·cnov + 0.20·chal), clipped to 0–100 and rounded to 1 dp. Return `None` if any input is missing or non-finite.
    - In `score_submission`, set `result["divergence_score_d"]` from the ungated `semantic_divergence`, `details.claim_novelty` and `details.assumption_challenge`. All existing fields and the function signature stay unchanged.
    - Bump `VERSION` to `diverge-score-0.2`. Document the D formula in the module docstring and mark it "experimental, NOT validated".
    - _Requirements: 1.6, 11.1, 11.2, 11.3, 11.8_
  - [ ] 1.2 Create `diverge/config.py`
    - Add `EVAL_CONFIG`, mirroring the scoring constants and `value.DEFAULT_CONFIG` by import. Set entailment_threshold 0.7, borderline_margin 0.1, ivs_threshold 45, d_weights, model ids, per-device precision and seed 7.
    - Add `canonical()`, `config_hash()` (sha256), `validate()` (raises ValueError naming each bad field) and `environment()`.
    - _Requirements: 4.6, 4.11, 9.1, 9.2, 9.6, 9.7, 9.8_
  - [ ]* 1.3 Pin `hypothesis` in `requirements.txt` (a Python 3.9-compatible version, e.g. `hypothesis==6.122.3`, confirmed at install). Register the `diverge` profile (max_examples=100, derandomize=True) in `tests/conftest.py`. Create `tests/test_properties.py` with properties 5 (D separation) and 9 (config hash round trip)
    - **Property 5: Separation of signals**
    - **Property 9: Config hash round trip**
    - **Validates: Requirements 10.1, 10.11, 11.2, 9.7**

- [ ] 2. Backend: verifiable questions and correctness checker
  - [ ] 2.1 Add q11–q14 to `diverge/questions.py` with `verifiable=True`
    - Cover bat-and-ball (asks for the answer "in cents"), Monty Hall, the base-rate false-positive problem and the birthday paradox. q01–q10 stay unchanged.
    - _Requirements: 14.1_
  - [ ] 2.2 Create `data/references.json` with only the objective keys for q11–q14
    - Use the Expected_Answer values, tolerances, known-wrong values and citations from the design table. Each key gets one Accepted_Claim and one Rejected_Claim, both taken from the cited source's standard statement.
    - Add a top-level `"_TODO"` note saying that open-ended keys for q01–q10 must be written by a human (task 2.5). The loader skips keys that start with `_`.
    - Do not generate any claim with an LLM.
    - _Requirements: 2.2, 14.2_
  - [ ] 2.3 Create `diverge/correctness.py`
    - Add `Checker` (loads the file once and collects load warnings without raising), `validate_key`, `extract_answer` and `match_expected`.
    - The Expected_Answer path runs first. If NLI fails, supplementary evidence is `[]` and the decision stands.
    - The NLI-claims path uses premise = claim. It checks incorrect, then correct, then unverified, and records the best pair as evidence.
    - With no key, return "no reference key" without calling NLI. On an NLI exception, return "verification unavailable".
    - `check()` takes no baselines argument.
    - Add a `python -m diverge.correctness --check` CLI.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1–3.9, 5.7, 14.3, 14.4, 14.5, 14.7_
  - [ ]* 2.4 Add model-free tests in `tests/test_correctness.py` for every shipped Expected_Answer (correct value, each known-wrong value, no number) and for loader edge cases (missing file, invalid JSON, malformed key, unknown id), plus properties 11, 12 and 13 in `tests/test_properties.py`
    - **Property 11: Expected-answer matching**
    - **Property 12: Correctness ignores baselines**
    - **Property 13: Reference-key validation**
    - **Validates: Requirements 2.2–2.5, 3.9, 10.6, 10.12, 14.3, 14.4, 14.7**
  - [ ]* 2.5 Add human-authored open-ended keys for at least 3 of q01–q10 (candidates: q02, q04, q05) to `data/references.json`
    - The user or developer writes the claims and citations. Nothing is drafted by an LLM.
    - Run `.venv/bin/python -m diverge.correctness --check` until there are 0 load warnings and at least one Accepted_Claim contradicts a cached consensus assumption at ≥ 0.7.
    - Remove the `_TODO` note.
    - _Requirements: 2.6, 2.7_

- [ ] 3. Backend: offline baselines for q11–q14
  - [ ] 3.1 Create `diverge/gen_baselines.py`, which calls `baselines.get_baselines(BY_ID[qid]["text"])` for q11–q14 and skips any question already cached. Run it once with `.venv/bin/python -m diverge.gen_baselines` (about 20 s per question) so `data/baselines.json` gains the entries without changing the existing ones
    - _Requirements: 14.8, 14.9_

- [ ] 4. Backend: evaluation pipeline
  - [ ] 4.1 Create `diverge/evaluate.py`
    - `evaluate()` runs these steps in order: validate config, seed everything, join reasoning and evidence, call `score_submission`, run the checker (on exception, status `unverified` plus a warning), call `value.intelligence_value`, compute uncertainty, then build the `assessment` dict with hashes and environment.
    - Guard the case of 0 baselines.
    - Add the pure helpers `breakdown()`, `interval_from()` (leave-one-out plus the correctness swap when unverified, the width rule, "borderline verification" and "insufficient baselines"), `explain_v2()` and `training_grade()`.
    - Add `assess_legacy()`, which builds an in-memory view: IVS from stored components using `unverified`, D/interval/breakdown set to None, and `not_rescored=True`.
    - _Requirements: 1.1–1.5, 1.7, 4.1–4.5, 4.12, 5.1–5.5, 6.1–6.8, 7.1–7.5, 7.7, 7.8, 9.2, 9.3, 9.6, 11.7, 13.1, 13.2, 13.5, 14.6, 17.1, 17.2_
  - [ ]* 4.2 Add properties 1, 2, 3, 4, 6, 7, 8 and 10 to `tests/test_properties.py` (model-free; Property 10 stubs `models.embed/nli/perplexity`)
    - **Property 1: Novelty is a bonus, never a penalty**
    - **Property 2: Incorrect answers are capped and never training-grade**
    - **Property 3: Correct conventional answers reach the floor**
    - **Property 4: Correctness ordering**
    - **Property 6: Ranges and validation**
    - **Property 7: Interval contains the score**
    - **Property 8: Breakdown sums to the pre-rule IVS**
    - **Property 10: Metadata does not affect scores**
    - **Validates: Requirements 4.3–4.12, 5.4, 5.5, 6.1–6.4, 7.1–7.4, 7.7, 7.8, 10.2–10.4, 10.10, 13.1, 17.3**

- [ ] 5. Backend: store additions
  - [ ] 5.1 Extend `diverge/store.py` (additive only)
    - Make the path overridable with `DIVERGE_SUBMISSIONS`.
    - Make `add()` serialize first and then do a single `write()`, truncating back to the original size on error.
    - Add `view()`, which uses `assess_legacy` and the "not rescored" / "invalid status" flags.
    - Add the new `table()` columns, a `training_grade` count in `leaderboard()` and a new `impact_leaderboard()`.
    - Add the export fields: status, verification, ivs, ivs_low/high, uncertainty, D and `not_rescored` (in `metadata` for the chat and DPO formats). The legacy `divergence_score` stays as is, and so does `TRAINING_THRESHOLD`.
    - _Requirements: 8.2, 8.5–8.10, 11.6, 11.7, 12.2, 13.1–13.3, 13.5, 13.6, 18.5_
  - [ ]* 5.2 Add Property 14 and legacy tests in `tests/test_store.py`, using a `tmp_path` copy and `DIVERGE_SUBMISSIONS`
    - Check that all 11 records load via `view`, that `table` and `leaderboard` build, and that all 3 exports parse.
    - Check that the sha256 of the real file is unchanged.
    - **Property 14: Export round trip and legacy safety**
    - **Validates: Requirements 8.5–8.7, 8.10, 10.13, 11.7, 13.5, 13.6**

- [ ] 6. Checkpoint: backend
  - Run `.venv/bin/python -m pytest`. The existing 24 tests in `tests/test_scoring.py` must pass unchanged.
  - Run `.venv/bin/python -m diverge.seed` as a dry check against a temp copy (`DIVERGE_SUBMISSIONS=/tmp/...`) and confirm it scores without errors.
  - Confirm the sha256 of `data/submissions.jsonl` is unchanged and the current app on :8501 still loads.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. App: four-page redesign in `app.py` (reuses `card`, `style`, `gauge`, `radar`, `contributions_bar` and the CSS)
  - [ ] 7.1 Add the router and HOME page
    - Set the background to charcoal `#0B0A10` and keep the purple and lime palette.
    - Add a nav with four `st.button`s (HOME, CHALLENGES, MY IMPACT, ABOUT) and `st.session_state.page`/`challenge`.
    - Add a `badge(status, flag)` helper that uses text labels.
    - HOME shows the hero copy, "Start a Challenge" (navigates) and "How It Works" (expands), the four step cards and the climate chain labeled "hypothesis".
    - Add an experiment summary of up to 4 values with provenance from `experiment.summarize`/`significance`, or "pending". No tables.
    - _Requirements: 15.1–15.10, 20.1, 20.3, 20.4_
  - [ ] 7.2 Add the CHALLENGES list and Challenge_Workspace
    - The list shows rows with title, difficulty, domain, Verification_Type and Solved (per Display_Name), plus an Open button.
    - Left column: statement, "AI baseline approach", "assumptions to investigate (hypotheses, not errors)", a baselines expander and the Consensus Breaker expander. Use `generate_if_missing=False` and show "baselines not cached".
    - Right column: name, answer, reasoning, evidence, consent (default False), the verifiable hint and "Submit for Evaluation".
    - Validate inputs with field-specific errors and keep the entered text.
    - On submit, call `evaluate()` and then `store.add()`. A store error shows "not saved". An IVS error shows D plus the legacy composite and stores nothing.
    - _Requirements: 8.9, 12.1, 12.5, 12.6, 12.8, 14.10, 15.2, 16.1–16.7, 16.12, 20.2_
  - [ ] 7.3 Add the result view and "How scoring works"
    - Show three cards: D ("originality only, not a quality measure"), Q (estimate) and IVS ("Estimated Learning Value (estimate)") with low/high and level.
    - Show the badge with "correctness not checked against a human reference" when unverified, and "capped at 25" when incorrect.
    - Show up to 4 explanation sentences.
    - Add an expander with the breakdown table, the cap/floor record, all sentences, Marginal_Value labeled "estimated proxy, not measured learning gain" and the legacy gauge/radar labeled "legacy composite". Show "no breakdown, not rescored" or "not rescored" for legacy records.
    - All values are read from the dicts with no maths in the UI.
    - Add the "How scoring works" copy, including that only automated local evaluators are used and there is no human panel.
    - _Requirements: 6.9, 6.10, 7.6, 7.9, 8.1, 11.4, 11.5, 11.7, 12.3, 12.4, 12.7, 16.8–16.11, 17.5, 17.6_
  - [ ] 7.4 Add the MY IMPACT page and Export_Panel
    - Show the Display_Name, the counts, average D ("how different, not how good") and average IVS (estimate), or "no data".
    - Add the "Your contribution to greener AI" pathway with the user's counts and "not yet measured", plus a note that per-user energy can't be attributed. No kWh is shown.
    - Add `impact_leaderboard` with "demo" labels.
    - Add the Export_Panel: 3 status toggles (correct/unverified on, incorrect off), a min-IVS slider defaulting to 45, gate and demo filters, a consent note, a disabled button at 0 selected, export of selected ids only and the 3 formats.
    - Read the store on every rerun.
    - _Requirements: 8.3, 8.4, 13.3, 13.4, 18.1–18.6, 19.10_
  - [ ] 7.5 Add the ABOUT page
    - Show Vision, Mission, Goal and Belief verbatim, plus the methodology prose.
    - "Measured performance" shows the existing experiment charts and tables from `results.json` with config, seeds, dataset, model and date, plus the null-result sentence, or "pending" with the run command.
    - "Estimated energy impact" shows only analytic FLOPs and wall-clock from `results.json`.
    - "Hypothetical scenarios" holds the Energy_Calculator: kWh = kW × h, scenario h = h × (1 − r), r defaults to 0%, and each input is labeled "assumption". Add the rebound, utilization and inference caveat.
    - Update the Method table with D and IVS.
    - _Requirements: 19.1–19.9_
  - [ ]* 7.6 Add AppTest smoke tests in `tests/test_app.py` with an autouse fixture that copies `submissions.jsonl` to `tmp_path` and sets `DIVERGE_SUBMISSIONS`
    - For each of the four pages, check there is no exception and that the key copy is present (the energy default is 0 kWh savings).
    - Add a validation case where a name containing "@" gives an error and no line is written.
    - _Requirements: 15.1, 15.3, 16.1, 16.6, 18.3, 19.6, 19.7_

- [ ] 8. Model-backed tests
  - [ ] 8.1 Add `tests/test_correctness_models.py` (marked `models`, skips with the missing model name if the local cache is absent)
    - Cover fixtures (a)–(d), the edge cases, a determinism test that scores twice, the Req 2.7 contradiction check (skipped until task 2.5 is done), and a check that `demo:noise`/`demo:offtopic` are not training-grade.
    - Add the AppTest submission flow: open q12, submit "2/3 because…" with consent, then assert the "Verified correct" badge and the D/IVS labels, that the temp file has 12 lines, and that the real file's hash is unchanged.
    - _Requirements: 2.8, 10.5–10.9, 10.14, 17.7_

- [ ] 9. Final checkpoint
  - Run `.venv/bin/python -m pytest`. The existing 24 tests and all new tests must pass, and model tests may skip only for a missing cache.
  - Run `.venv/bin/python -m diverge.seed` against a temp `DIVERGE_SUBMISSIONS` copy and confirm there are no errors.
  - Confirm the sha256 of `data/submissions.jsonl` matches the value recorded at the start and that the 11 legacy records load and render in MY IMPACT and the Export_Panel.
  - Confirm the :8501 server hot-reloaded without errors.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Deferred: Req 9 rescore flow
  - [ ]* 10.1 Add `rescore(record)` to `diverge/evaluate.py`, returning new values as a separate result with the "config changed", "environment changed" and "not reproducible" flags. Do not modify the stored record
    - _Requirements: 9.4, 9.5, 9.9, 9.10_
  - [ ]* 10.2 Write unit tests for `rescore` flags in `tests/test_rescore.py`
    - _Requirements: 9.4, 9.5, 9.9, 9.10_

## Notes

- Tasks marked `*` are optional: Hypothesis property tests, model-free unit tests, AppTest smoke tests, human-authored open-ended keys (2.5), and the deferred rescore flow.
- Task 2.5 must be done by a human. Until then, open-ended questions stay `unverified` and the Req 2.6/2.7 checks remain open.
- Never start a second Streamlit server. Edits to `app.py` hot-reload on :8501.
- Tests never touch the real `data/submissions.jsonl`. They use `DIVERGE_SUBMISSIONS` pointed at a temp copy.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "3.1"] },
    { "id": 2, "tasks": ["2.3", "1.3"] },
    { "id": 3, "tasks": ["4.1"] },
    { "id": 4, "tasks": ["5.1", "2.4"] },
    { "id": 5, "tasks": ["7.1", "4.2"] },
    { "id": 6, "tasks": ["7.2", "5.2", "2.5"] },
    { "id": 7, "tasks": ["7.3", "10.1"] },
    { "id": 8, "tasks": ["7.4", "10.2"] },
    { "id": 9, "tasks": ["7.5"] },
    { "id": 10, "tasks": ["7.6", "8.1"] }
  ]
}
```
