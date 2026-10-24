# Implementation Plan: climate-intelligence-engine

## Overview

Build `climate_intelligence/` as a smallest-working vertical slice, in Python 3.9 (`from __future__ import annotations` in every module). The order is valuation → impact → lab → schemas/API → demo/README → optional server. Every task creates new files only.

Constraints for every task:
- Never modify existing files, `engine/`, `unbound/`, `data/experiment/results.json`, or `data/submissions.jsonl`.
- No dependency installs. `jsonschema` 4.25.1 is already in `.venv`. `hypothesis` is not installed, so property tests are optional and begin with `pytest.importorskip("hypothesis")`.
- No long experiments. Tests use tiny synthetic data (n_train ≤ 200, epochs ≤ 2). The real run is 1 seed × 1 budget.
- Never start Streamlit. One server is already running on :8501.
- Tests write Lab_Results to `tmp_path`, never to `data/climate_intelligence/`.
- Test command: `.venv/bin/python -m pytest tests/climate_intelligence tests/test_scoring.py -q`

Property tests go in one file per property, `tests/climate_intelligence/test_prop_NN_<name>.py`, instead of the single `test_properties.py` named in the design. This keeps parallel tasks from writing to the same file. Each file is tagged `# Feature: climate-intelligence-engine, Property NN: <title>` and uses `@settings(max_examples=100, deadline=None)`.

## Tasks

- [ ] 1. Package skeleton and status primitives
  - [ ] 1.1 Create `climate_intelligence/__init__.py` and `climate_intelligence/status.py`
    - `__init__.py`: `VERSION = "cie-0.1-experimental"`. API re-exports are added in 8.2.
    - `status.py`:
      - `STATUSES` ordered strongest to weakest, and `weakest(*statuses)`.
      - `provenance(value, unit, status, source, assumption=False, formula=None)`.
      - `CIError(code, message, field=None)` with `.to_dict()` returning an Error_Object.
    - _Requirements: 10.1, 12.3_
  - [ ] 1.2 Write unit tests in `tests/climate_intelligence/test_status.py`
    - Test the `weakest` ordering, the provenance keys, and the `CIError.to_dict()` shape.
    - _Requirements: 10.1, 12.3_

- [ ] 2. Correctness and valuation (Module 1)
  - [ ] 2.1 Implement `climate_intelligence/correctness.py`
    - `validate_reference`: raises a ValueError naming `source` or `reference_id`.
    - `check_correctness(answer_text, references=None, human_label=None)`:
      - Normalizes text and matches `accepted` aliases (exact or whole-word), `numeric` values within rel_tol, and `rejected` entries.
      - A `human_label` takes precedence.
      - Below `MATCH_THRESHOLD`, or with no references, returns `unverified` with a reason.
      - Records `reference_id` and `source` on a verified result.
      - Has no baseline parameter.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  - [ ] 2.2 Implement `climate_intelligence/valuation.py`
    - `value(component_scores, correctness_status=None, *, answer_text, references, human_label, config)`:
      - Validates the four scores first and raises `ValueError(<field>)` for missing, non-numeric, bool, NaN/inf, or out-of-range values.
      - Takes the status from the explicit argument or from `check_correctness`.
      - Calls `diverge.value.combine` and labels training_value `estimated`, with `rule_applied` and `ivs_version`.
      - Sets eligibility: `False` for `verified_incorrect`, `"needs_review"` for `unverified`, and `ivs >= IVS_THRESHOLD` for `verified_correct`.
      - Sets the unverified reason to mention the missing human-authored reference.
      - Returns a JSON-safe dict.
    - `score_with_models(question, answer, baselines)`:
      - Lazily imports `diverge.models` and `diverge.scoring`, and maps output via `diverge.value.dimensions`.
      - Any load failure raises `CIError("models_unavailable")`.
      - Baselines go only to `score_submission`.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 3.1, 3.2, 3.3, 3.4_
  - [ ] 2.3 Write unit tests in `tests/climate_intelligence/test_correctness.py`
    - Cover alias match, rejected match, numeric match, and no-match → `unverified`.
    - Cover a missing `source` (ValueError naming `source`) and `human_label` precedence.
    - Check with `inspect.signature` that there is no `baselines` parameter.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  - [ ] 2.4 Write unit tests in `tests/climate_intelligence/test_valuation.py`
    - Three core cases:
      - Incorrect-but-novel (novelty 95): value ≤ 25 and ineligible.
      - Correct-but-conventional (novelty 0, high reasoning): value ≥ 55.
      - Unverified: `needs_review`, method `none`.
    - Each bad-score kind raises a ValueError naming the field.
    - The JSON round trip gives an equal dict.
    - `diverge.models` is not in `sys.modules` after `value()`.
    - `models_unavailable` is raised (via monkeypatch). The real adapter test skips if the models are not cached.
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 3.1, 3.3, 14.2_
  - [ ]* 2.5 Property test: valuation shape, labels and model agreement
    - **Property 1: Valuation shape, labels and model agreement**
    - **Validates: Requirements 1.1, 1.2, 1.3**
  - [ ]* 2.6 Property test: incorrect cap
    - **Property 2: Incorrect cap**
    - **Validates: Requirements 1.4**
  - [ ]* 2.7 Property test: conventional floor
    - **Property 3: Conventional floor**
    - **Validates: Requirements 1.5**
  - [ ]* 2.8 Property test: unverified handling
    - **Property 4: Unverified handling**
    - **Validates: Requirements 1.6, 2.2**
  - [ ]* 2.9 Property test: valuation JSON round trip
    - **Property 5: Valuation JSON round trip**
    - **Validates: Requirements 1.7**
  - [ ]* 2.10 Property test: invalid component scores rejected
    - **Property 6: Invalid component scores rejected**
    - **Validates: Requirements 1.8**
  - [ ]* 2.11 Property test: novelty independence
    - **Property 7: Novelty independence**
    - **Validates: Requirements 1.9**
  - [ ]* 2.12 Property test: reference matching decided and recorded by references only
    - **Property 8: Reference matching is decided by references only, and recorded**
    - **Validates: Requirements 2.1, 2.4, 2.6**
  - [ ]* 2.13 Property test: references without source rejected
    - **Property 9: References without source rejected**
    - **Validates: Requirements 2.3**

- [ ] 3. Checkpoint: valuation slice
  - Run `.venv/bin/python -m pytest tests/climate_intelligence tests/test_scoring.py -q`. Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Energy and emissions calculator (Module 3)
  - [ ] 4.1 Implement `climate_intelligence/impact.py`
    - `compute_energy(...)`:
      - Validates inputs and raises `CIError("invalid_input", field)` for each of the 4 fields.
      - Raises `missing_power` when there is no power and no `assumed_power_w`.
      - Computes `it_kwh = W*h/1000`, `facility_kwh = it_kwh*PUE`, and `kg_co2e = facility_kwh*GI`.
      - Attaches a Provenance_Record with a formula to every input and output.
    - Status rules:
      - Each output gets the weakest input status.
      - Assumptions cap output status at `estimated`.
      - Any synthetic input makes the output `synthetic`.
      - Any `extrapolation` block is tagged `hypothetical`.
      - Every result carries the `scope` string.
    - `compare_arms(lab_result, baseline_arm, comparison_arm, budget, **energy_inputs)`:
      - Uses mean `train_seconds/3600` per arm as runtime.
      - Inherits the row status (`measured` or `synthetic`).
      - Computes `difference` as comparison − baseline.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 11.1, 11.2, 11.3, 11.4_
  - [ ] 4.2 Write unit tests in `tests/climate_intelligence/test_impact.py`
    - Formulas on a hand-computed example.
    - Each invalid field, and `missing_power`.
    - An assumed power value is never `measured`, and a synthetic input propagates.
    - The `scope` field is present and extrapolation is `hypothetical`.
    - `compare_arms` on a hand-built Lab_Result dict, with the sign of the difference.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 11.1, 11.2, 11.3, 11.4, 14.3_
  - [ ]* 4.3 Property test: energy formulas and provenance
    - **Property 19: Energy formulas and provenance**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.5, 9.6**
  - [ ]* 4.4 Property test: scenario difference
    - **Property 20: Scenario difference**
    - **Validates: Requirements 9.4**
  - [ ]* 4.5 Property test: conservative status propagation
    - **Property 21: Conservative status propagation**
    - **Validates: Requirements 10.1, 10.2, 10.4**
  - [ ]* 4.6 Property test: calculator input validation
    - **Property 22: Calculator input validation**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4**

- [ ] 5. Checkpoint: impact slice
  - Run `.venv/bin/python -m pytest tests/climate_intelligence tests/test_scoring.py -q`. Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Compute-efficiency lab (Module 2)
  - [ ] 6.1 Implement `climate_intelligence/lab/data.py`
    - `Dataset` dataclass.
    - `load_real()`:
      - Sets `HF_HUB_OFFLINE=1` and reads `data/experiment/csqa_embeddings.npz` read-only.
      - Raises `CIError("data_unavailable")` if the npz is missing. The path is injectable for tests.
      - Takes tokens and text hashes from the offline CommonsenseQA parquet, with `whitespace_words_q_plus_choices`.
      - Falls back to `constant_1_fallback` tokens with `embedding_bytes` hashes.
    - `make_synthetic(seed, n_train, n_eval, d)`: deterministic.
    - _Requirements: 5.2, 7.1, 7.3_
  - [ ] 6.2 Implement `climate_intelligence/lab/arms.py`
    - `budget_fill` stops at the first example that would exceed the budget.
    - `select_random`, `select_novelty`, `typicality_gate`, and `select_gated`:
      - `select_novelty` uses farthest-point ranking.
      - `typicality_gate` uses a train-only centroid with p10/p60 bounds.
      - `select_gated` ranks by k-center distance × gate.
    - The functions accept training arrays only.
    - _Requirements: 4.1, 4.2, 4.4, 4.5, 5.4_
  - [ ] 6.3 Implement `climate_intelligence/lab/leakage.py`
    - `check_leakage(train_ids, eval_ids, train_hashes, eval_hashes)`:
      - Returns `{id_overlap, hash_overlap, hash_basis, passed}`.
      - Raises `CIError("leakage_detected")` with the overlap count in the message.
    - _Requirements: 5.1, 5.2, 5.3, 5.5_
  - [ ] 6.4 Implement `climate_intelligence/lab/stats.py`
    - `summarize(rows)`.
    - `significance(rows, a, b)`: paired sign-flip test, 10k permutations, `rng(0)`, exact enumeration when `2^s ≤ 10k`.
    - `min_attainable_p`, and a `summary_statement` of `"no significant difference"` when all p ≥ 0.05.
    - _Requirements: 6.6, 6.7_
  - [ ] 6.5 Implement `run_lab` in `climate_intelligence/lab/__init__.py`
    - Setup and run order:
      - Merge `DEFAULT` with the config. Raise `ValueError("token_budget")` when the budget exceeds the pool total.
      - Run the leakage check before training.
      - Override `diverge.experiment.CONFIG` in a context manager and restore it in `finally`.
      - Call `diverge.experiment.train_eval` with identical hyper-parameters and seeds for every arm.
    - Per-run rows:
      - Record tokens, steps, wall and CPU seconds, and estimated FLOPs.
      - Record accuracy and accuracy above chance per PFLOP.
      - Record `gpu_hours` (`measured` on CUDA, otherwise null with `no_gpu`), and null `measured_energy_kwh` with a reason.
    - Result fields:
      - Add a `budget_mismatch` warning when needed.
      - Synthetic mode: every status is `synthetic` and `evidence` is false.
      - Include environment, schema_version, and completed_at.
    - Output:
      - Write to `out_dir/lab_<mode>_<UTC>.json` with a `_N` suffix on collision. Never touch `results.json`.
    - _Requirements: 4.3, 4.6, 4.7, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 7.2, 7.4, 7.5, 8.4_
  - [ ] 6.6 Write unit tests in `tests/climate_intelligence/test_lab.py`, synthetic mode only
    - Arms fill the budget, gated selection skips zero-gate examples, and novelty ignores the gate.
    - A `budget_mismatch` fixture, and the over-budget ValueError.
    - Synthetic tagging and `evidence: false`.
    - Determinism across two runs.
    - Selection indices are unchanged when eval data changes.
    - `diverge.experiment.CONFIG` is restored after an injected failure.
    - `data_unavailable` with a tmp npz path.
    - The SHA-256 of `results.json` is unchanged, and a file is written to `tmp_path`.
    - `gpu_hours` is null with `no_gpu`.
    - _Requirements: 4.1, 4.2, 4.4, 4.5, 4.6, 4.7, 5.4, 6.1, 6.3, 6.4, 6.5, 7.2, 7.3, 7.4, 7.5, 14.3_
  - [ ] 6.7 Write unit tests in `tests/climate_intelligence/test_leakage.py`
    - Disjoint sets pass. Injected id or hash overlap raises `leakage_detected` with the count.
    - _Requirements: 5.1, 5.2, 5.3, 14.3_
  - [ ] 6.8 Write the integration test in `tests/climate_intelligence/test_lab_real.py`
    - Skip if the npz is missing.
    - Run 1 seed × 1 budget in real mode with `out_dir=tmp_path`.
    - Assert `evidence: true`, leakage passed, and `results.json` unchanged.
    - _Requirements: 5.5, 7.1, 7.4_
  - [ ]* 6.9 Property test: budget fill
    - **Property 10: Budget fill**
    - **Validates: Requirements 4.2**
  - [ ]* 6.10 Property test: zero-gate examples never preferred
    - **Property 11: Zero-gate examples never preferred**
    - **Validates: Requirements 4.4**
  - [ ]* 6.11 Property test: budget mismatch warning
    - **Property 12: Budget mismatch warning**
    - **Validates: Requirements 4.6**
  - [ ]* 6.12 Property test: over-budget rejected
    - **Property 13: Over-budget rejected**
    - **Validates: Requirements 4.7**
  - [ ]* 6.13 Property test: leakage detection
    - **Property 14: Leakage detection**
    - **Validates: Requirements 5.1, 5.2, 5.3**
  - [ ]* 6.14 Property test: selection is eval-independent
    - **Property 15: Selection is eval-independent**
    - **Validates: Requirements 5.4**
  - [ ]* 6.15 Property test: significance well-formed
    - **Property 16: Significance well-formed**
    - **Validates: Requirements 6.6, 6.7**
  - [ ]* 6.16 Property test: synthetic tagging and determinism
    - **Property 17: Synthetic tagging and determinism**
    - **Validates: Requirements 7.2, 7.5**
  - [ ]* 6.17 Property test: Lab_Result schema and round trip
    - **Property 18: Lab_Result schema and round trip**
    - **Validates: Requirements 8.2, 8.3, 8.4**

- [ ] 7. Checkpoint: lab slice
  - Run `.venv/bin/python -m pytest tests/climate_intelligence tests/test_scoring.py -q`. Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Schemas and integration API (Module 4)
  - [ ] 8.1 Create `climate_intelligence/schemas/*.schema.json` (Draft 2020-12) and `climate_intelligence/schema.py`
    - Schemas: valuation_request, valuation_result, lab_request, lab_result, impact_request, impact_result, and error.
    - `schema.py`: `load_schema(name)` and `validate(obj, name)` using `jsonschema`.
    - _Requirements: 8.1, 8.4, 12.2_
  - [ ] 8.2 Implement `climate_intelligence/api.py` and add re-exports to `__init__.py`
    - Functions: `value_submission`, `run_lab`, `estimate_impact`.
    - Each validates the request, calls its module, and validates the response.
    - Errors:
      - `CIError` or `ValueError` → Error_Object.
      - Any other exception → `internal_error`, with no traceback in the message.
    - `estimate_impact` accepts either direct inputs or a Lab_Result arm comparison.
    - _Requirements: 12.1, 12.3, 12.4, 12.7_
  - [ ] 8.3 Create `climate_intelligence/examples/<fn>_request.json` and `<fn>_response.json` for the three functions
    - The lab example uses a tiny synthetic config.
    - Generate the responses from the actual code, never hand-write numbers.
    - _Requirements: 12.2, 12.5_
  - [ ] 8.4 Write `tests/climate_intelligence/test_schemas.py` and `tests/climate_intelligence/test_api.py`
    - Synthetic Lab_Result validates and round-trips.
    - Shipped example requests produce schema-valid responses.
    - Bad requests return Error_Objects.
    - `internal_error` is returned (via monkeypatch).
    - Calls run with network blocked: `socket.connect` is patched to reject non-loopback addresses.
    - _Requirements: 8.1, 8.2, 8.3, 12.1, 12.2, 12.3, 12.4, 12.5, 14.1, 14.3_
  - [ ]* 8.5 Property test: API never raises on bad requests
    - **Property 23: API never raises on bad requests**
    - **Validates: Requirements 12.3**

- [ ] 9. Checkpoint: API slice
  - Run `.venv/bin/python -m pytest tests/climate_intelligence tests/test_scoring.py -q`. Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Demo CLI and README (Module 5)
  - [ ] 10.1 Implement `climate_intelligence/demo.py`
    - `main(argv)` prints, in order:
      1. Valuation of sample submissions, including a fluent incorrect contrarian submission with a human reference.
      2. Quality-gated selection, showing the contrarian excluded.
      3. Compute comparison.
      4. Climate estimate, with power, PUE and GI shown as assumptions.
    - Every number goes through `fmt(value, status)`.
    - Synthetic mode (with a banner) runs on `--synthetic` or when the npz is missing.
    - A null-result line is printed when there is no significant difference.
    - Uses a small config. Writes Lab_Results via `--out-dir`, defaulting to `data/climate_intelligence/`.
    - Add `__main__` support.
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_
  - [ ] 10.2 Write `tests/climate_intelligence/test_demo.py`
    - Run `main(["--synthetic", "--out-dir", tmp_path])` and capture stdout.
    - Check section order, a status label before every number (regex), the contrarian excluded, the banner, and the null-result line. Run with network blocked.
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_
  - [ ] 10.3 Write `climate_intelligence/README.md`
    - Setup, CLI usage, API examples, and schema/example locations.
    - Status label definitions, quality-proxy and token-proxy caveats, `min_attainable_p`, and the null-result context.
    - The optional server, with a note that it has no authentication.
    - _Requirements: 12.6, 14.4_

- [ ] 11. Optional localhost HTTP wrapper
  - [ ]* 11.1 Implement `climate_intelligence/server.py`
    - `serve(port=8765, host="127.0.0.1")` serves POST `/value`, `/lab` and `/impact` with stdlib `ThreadingHTTPServer`.
    - Refuses non-loopback hosts. The docstring states it has no authentication.
    - _Requirements: 12.6_
  - [ ]* 11.2 Write `tests/climate_intelligence/test_server.py`
    - Start the server on 127.0.0.1 port 0 in a thread, POST `/impact` once, and shut it down.
    - A non-loopback host raises ValueError.
    - _Requirements: 12.6_

- [ ] 12. Final checkpoint
  - Run `.venv/bin/python -m pytest tests/climate_intelligence tests/test_scoring.py -q`.
  - Confirm with `git status` that no tracked file changed.
  - Ensure all tests pass, ask the user if questions arise.
  - Do not start Streamlit.
  - _Requirements: 14.5, 14.6_

## Notes

- Tasks marked `*` are optional: property tests (need `hypothesis`, auto-skipped) and the server.
- Real-mode and model-adapter tests skip cleanly when cached data or models are absent.
- All energy numbers come from caller-given power, which is marked as an assumption. No measured energy is fabricated.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "4.1", "6.1", "6.2", "6.3", "6.4", "8.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.12", "2.13", "4.2", "4.3", "4.4", "4.5", "4.6", "6.5", "6.7", "6.9", "6.10", "6.13", "6.14", "6.15"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "6.6", "6.8", "6.11", "6.12", "6.16", "6.17", "8.2"] },
    { "id": 4, "tasks": ["8.3", "8.5", "10.1", "11.1"] },
    { "id": 5, "tasks": ["8.4", "10.2", "10.3", "11.2"] }
  ]
}
```
