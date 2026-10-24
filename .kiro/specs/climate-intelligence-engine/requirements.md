# Requirements Document

## Introduction

`climate_intelligence/` is a standalone, local-only Python package (Python 3.9, `from __future__ import annotations`) at the repo root. It turns human reasoning submissions into an auditable chain: intelligence valuation → quality-gated data selection → compute-efficiency comparison → climate (energy / CO2e) estimate. Every number carries a status label (MEASURED, ESTIMATED, HYPOTHETICAL, SYNTHETIC) and a provenance record.

The package reuses `diverge.value.combine` (Incorrect_Cap 25, Conventional_Floor 55, validated config) and optionally `diverge.scoring.score_submission`. It does not modify existing files, `engine/`, `unbound/`, `data/experiment/results.json`, or the teammate repository (maudeniemann/divergence-for-data-centers). Integration with that repository is only through the documented Python functions and JSON Schemas defined here.

Known measured context: the existing CommonsenseQA proxy experiment found divergent selection beat conventional selection at small N but did not beat random selection, with no result significant at p < 0.05. The package must be able to reproduce and report results of this kind, including null results.

## Glossary

- **Package**: the `climate_intelligence` Python package.
- **Valuator**: Module 1 (`climate_intelligence.valuation`), computes the Valuation_Result for one submission.
- **Component_Scores**: precomputed 0-100 dimensions `semantic_novelty`, `reasoning_quality`, `evidence_quality`, `marginal_value` (same meaning as `diverge.value.dimensions`).
- **Reference_Answer**: a caller-provided, human-authored reference with `reference_id`, `text` or accepted labels, and `source`. AI baselines and LLM outputs are never Reference_Answers.
- **Correctness_Status**: one of `verified_correct`, `verified_incorrect`, `unverified`.
- **Verification_Status**: object stating Correctness_Status, the method (`reference_match`, `human_label`, `none`), the reference_id used (or null), and a reason string.
- **Training_Value**: the IVS from `diverge.value.combine`, labeled ESTIMATED.
- **Valuation_Result**: JSON-serializable dict holding the five separated signals (semantic divergence, reasoning quality, correctness, Training_Value, Verification_Status) plus version and provenance.
- **Scoring_Adapter**: optional function that calls `diverge.scoring.score_submission` and maps its output to Component_Scores.
- **Lab**: Module 2 (`climate_intelligence.lab`), the compute-efficiency experiment framework.
- **Arm**: a selection strategy: `A_random`, `B_novelty_only`, `C_quality_gated_value`.
- **Token_Budget**: the maximum total training tokens (sum of per-example Token_Count) each Arm may select.
- **Token_Count**: the number of tokens of one training example under the documented tokenizer or documented proxy recorded in the run config.
- **Run_Mode**: `real` (local run on cached CommonsenseQA embeddings in `data/experiment/csqa_embeddings.npz`) or `synthetic` (generated data for demonstration).
- **Lab_Result**: JSON document of one Lab run, conforming to the Lab_Result_Schema, written to a new file under `data/climate_intelligence/`.
- **Calculator**: Module 3 (`climate_intelligence.impact`), the energy and emissions calculator.
- **Measurement_Status**: one of `measured`, `estimated`, `hypothetical`, `synthetic`, ordered by strength `measured` > `estimated` > `hypothetical` > `synthetic`.
- **Provenance_Record**: per-value record of value, unit, Measurement_Status, source description, and whether the value is an assumption.
- **PUE**: power usage effectiveness, facility energy divided by IT energy, ≥ 1.0.
- **Grid_Intensity**: grid carbon intensity in kg CO2e per kWh.
- **API**: Module 4 (`climate_intelligence.api`), the documented integration functions.
- **Error_Object**: JSON dict `{"error": {"code": str, "message": str, "field": str|null}}`.
- **Demo**: Module 5 CLI, `python -m climate_intelligence.demo`.

## Requirements

### Requirement 1: Separated intelligence valuation

**User Story:** As a researcher, I want each submission's divergence, reasoning, correctness, training value and verification status reported separately, so that novelty is never mistaken for correctness.

#### Acceptance Criteria

1. WHEN Component_Scores and a Correctness_Status input are provided, THE Valuator SHALL return a Valuation_Result containing separate fields for semantic divergence, reasoning quality, correctness, Training_Value, and Verification_Status.
2. THE Valuator SHALL compute Training_Value by calling `diverge.value.combine` with the provided Component_Scores and Correctness_Status.
3. THE Valuator SHALL label Training_Value with Measurement_Status `estimated`.
4. WHEN the Correctness_Status is `verified_incorrect`, THE Valuator SHALL return a Training_Value at or below the configured Incorrect_Cap and set `training_eligible` to false.
5. WHEN the Correctness_Status is `verified_correct` and reasoning_quality is at or above the configured floor threshold, THE Valuator SHALL return a Training_Value at or above the configured Conventional_Floor for every semantic_novelty value from 0 to 100.
6. WHEN the Correctness_Status is `unverified`, THE Valuator SHALL set Verification_Status method to `none`, set `training_eligible` to `"needs_review"`, and include a reason string stating that no human-authored reference was available.
7. THE Valuator SHALL produce a Valuation_Result that round-trips through `json.dumps` and `json.loads` to an equal dict.
8. IF a Component_Score is missing, non-numeric, non-finite, or outside 0-100, THEN THE Valuator SHALL raise a ValueError naming the offending field.
9. FOR ALL Component_Scores and Correctness_Status values, changing only semantic_novelty SHALL leave the reported correctness and Verification_Status unchanged (independence property).

### Requirement 2: Correctness from human-authored references only

**User Story:** As a researcher, I want correctness decided only by human-authored references, so that agreement with AI is never treated as truth.

#### Acceptance Criteria

1. WHEN a submission and a list of Reference_Answers are provided, THE Valuator SHALL determine Correctness_Status only from those Reference_Answers or from an explicit human label.
2. IF no Reference_Answer and no human label are provided, THEN THE Valuator SHALL set Correctness_Status to `unverified`.
3. IF a Reference_Answer lacks a `source` field, THEN THE Valuator SHALL reject the Reference_Answer with a ValueError naming `source`.
4. WHEN the reference match is below the documented confidence threshold, THE Valuator SHALL set Correctness_Status to `unverified` with a reason string.
5. THE Valuator SHALL accept no AI baseline input in the correctness function signature.
6. WHEN a verified Correctness_Status is set, THE Valuator SHALL record the `reference_id` and `source` used in Verification_Status.

### Requirement 3: Optional model adapter

**User Story:** As a developer, I want the valuation to run without heavy models but use them when present, so that tests stay fast and the real pipeline still works.

#### Acceptance Criteria

1. THE Valuator SHALL compute a Valuation_Result from precomputed Component_Scores without importing `diverge.models`.
2. WHERE local models are available, THE Scoring_Adapter SHALL call `diverge.scoring.score_submission` and map the result to Component_Scores using `diverge.value.dimensions`.
3. IF the local models cannot be loaded, THEN THE Scoring_Adapter SHALL raise an error with code `models_unavailable`.
4. THE Scoring_Adapter SHALL pass AI baselines only to `score_submission` and to no correctness function.

### Requirement 4: Comparable selection arms under a token budget

**User Story:** As a researcher, I want random, novelty-only and quality-gated selection compared under identical conditions, so that any difference is attributable to selection.

#### Acceptance Criteria

1. THE Lab SHALL implement the Arms `A_random`, `B_novelty_only`, and `C_quality_gated_value`.
2. THE Lab SHALL select examples for every Arm until the next example would exceed the Token_Budget, and record the selected total tokens per Arm.
3. THE Lab SHALL train every Arm with the same model architecture, optimizer, hyper-parameters, epochs, seeds and held-out evaluation set, recorded once in the run config.
4. THE `C_quality_gated_value` Arm SHALL rank examples by a novelty score multiplied by the quality gate, so that examples with zero quality gate have zero selection value.
5. THE `B_novelty_only` Arm SHALL rank examples by novelty score without the quality gate.
6. WHEN the selected total tokens of any two Arms differ by more than the size of the largest single example, THE Lab SHALL record a `budget_mismatch` warning in the Lab_Result.
7. IF the Token_Budget exceeds the total tokens of the training pool, THEN THE Lab SHALL raise a ValueError naming `token_budget`.

### Requirement 5: Leakage prevention

**User Story:** As a reviewer, I want train and eval data provably disjoint, so that performance numbers are not inflated.

#### Acceptance Criteria

1. WHEN a Lab run starts, THE Lab SHALL verify that training and evaluation example ids are disjoint.
2. WHEN a Lab run starts, THE Lab SHALL verify that the SHA-256 hashes of normalized training and evaluation text (or, where text is unavailable, of the embedding bytes) are disjoint.
3. IF any id or hash overlap is found, THEN THE Lab SHALL abort with an error with code `leakage_detected` listing the overlapping count.
4. THE Lab SHALL pass only training-pool data to the selection functions.
5. THE Lab SHALL record the leakage check results in the Lab_Result.

### Requirement 6: Compute and performance tracking

**User Story:** As a climate-tech evaluator, I want compute and performance recorded with honest status, so that efficiency claims are traceable.

#### Acceptance Criteria

1. THE Lab SHALL record per run: training tokens, optimizer steps, wall-clock seconds, CPU seconds, analytic training FLOPs, held-out accuracy, and accuracy above chance per PFLOP.
2. WHILE a GPU is used, THE Lab SHALL record GPU-hours with Measurement_Status `measured`.
3. IF no GPU is used, THEN THE Lab SHALL record GPU-hours as null with reason `no_gpu`.
4. IF no energy instrumentation is available, THEN THE Lab SHALL record measured energy as null with a reason string.
5. THE Lab SHALL label analytic FLOPs with Measurement_Status `estimated`.
6. THE Lab SHALL report per-budget mean accuracy difference between each pair of Arms with a paired permutation p-value over seeds.
7. WHEN no Arm difference has p < 0.05, THE Lab SHALL include the summary statement `no significant difference` in the Lab_Result.

### Requirement 7: Real and synthetic run modes

**User Story:** As a hackathon presenter, I want a real local run and a clearly marked synthetic demo, so that demonstration data is never shown as evidence.

#### Acceptance Criteria

1. WHERE Run_Mode is `real`, THE Lab SHALL load training and evaluation data from `data/experiment/csqa_embeddings.npz` without network access.
2. WHERE Run_Mode is `synthetic`, THE Lab SHALL tag every result value with Measurement_Status `synthetic` and set `evidence: false` in the Lab_Result.
3. IF Run_Mode is `real` and the embeddings cache is missing, THEN THE Lab SHALL raise an error with code `data_unavailable`.
4. THE Lab SHALL write each Lab_Result to a new file under `data/climate_intelligence/` and leave `data/experiment/results.json` unmodified.
5. WHEN the same config and seeds are run twice in the same environment, THE Lab SHALL produce identical selections and identical accuracies.

### Requirement 8: Lab result format

**User Story:** As an integrator, I want results in a documented JSON format, so that the teammate's system can consume them.

#### Acceptance Criteria

1. THE Package SHALL ship a JSON Schema file for Lab_Result.
2. THE Lab SHALL produce Lab_Results that validate against the Lab_Result_Schema.
3. FOR ALL Lab_Results, writing to JSON then loading SHALL produce an equal object (round-trip property).
4. THE Lab_Result SHALL include config, environment, Run_Mode, leakage check, per-run rows, summary, significance, and schema version.

### Requirement 9: Auditable energy and emissions calculation

**User Story:** As a climate-tech evaluator, I want energy and CO2e computed from explicit formulas and assumptions, so that every number can be audited.

#### Acceptance Criteria

1. WHEN power in W and runtime in hours are provided, THE Calculator SHALL compute IT energy kWh as power × hours / 1000.
2. THE Calculator SHALL compute facility energy kWh as IT energy kWh × PUE.
3. THE Calculator SHALL compute emissions kg CO2e as facility energy kWh × Grid_Intensity.
4. WHEN a baseline Lab_Result arm and a comparison Lab_Result arm are provided, THE Calculator SHALL compute the scenario difference in IT kWh, facility kWh and kg CO2e as comparison minus baseline.
5. THE Calculator SHALL return a Provenance_Record for every input and output value, including the formula used for each output.
6. FOR ALL valid inputs, facility kWh SHALL be greater than or equal to IT kWh and all outputs SHALL be non-negative except scenario differences (invariant property).

### Requirement 10: Measurement status propagation

**User Story:** As a reviewer, I want status labels to propagate conservatively, so that assumptions never appear as measurements.

#### Acceptance Criteria

1. THE Calculator SHALL assign each output the weakest Measurement_Status among the inputs used to compute the output.
2. WHEN an input is an assumption (PUE, Grid_Intensity, or default power), THE Calculator SHALL cap the status of every dependent output at `estimated`.
3. IF power in W is not provided, THEN THE Calculator SHALL use a caller-provided assumed power marked as an assumption, or raise an error with code `missing_power`.
4. WHEN any input has status `synthetic`, THE Calculator SHALL label every dependent output `synthetic`.
5. THE Calculator SHALL include a `scope` field stating that results describe the measured experiment only and do not represent data-center-wide savings.
6. THE Calculator SHALL label any extrapolation beyond the measured experiment as `hypothetical`.

### Requirement 11: Calculator input validation

**User Story:** As an integrator, I want clear errors for bad inputs, so that bad numbers never propagate silently.

#### Acceptance Criteria

1. IF power in W is negative, non-finite or non-numeric, THEN THE Calculator SHALL raise an error with code `invalid_input` and field `gpu_power_w`.
2. IF runtime hours is negative, non-finite or non-numeric, THEN THE Calculator SHALL raise an error with code `invalid_input` and field `runtime_hours`.
3. IF PUE is below 1.0 or non-finite, THEN THE Calculator SHALL raise an error with code `invalid_input` and field `pue`.
4. IF Grid_Intensity is negative or non-finite, THEN THE Calculator SHALL raise an error with code `invalid_input` and field `grid_intensity_kg_per_kwh`.

### Requirement 12: Integration API

**User Story:** As the teammate integrating DIVERGE into a data-center tool, I want documented functions and schemas, so that I can call the engine without knowing its internals.

#### Acceptance Criteria

1. THE API SHALL expose `value_submission(request: dict) -> dict`, `run_lab(request: dict) -> dict`, and `estimate_impact(request: dict) -> dict`, each accepting and returning JSON-serializable dicts.
2. THE Package SHALL ship JSON Schemas for the request and response of each API function and one example request and response per function.
3. WHEN a request fails validation, THE API SHALL return an Error_Object instead of raising.
4. WHEN an internal error occurs, THE API SHALL return an Error_Object with code `internal_error`.
5. FOR ALL example requests shipped with the Package, THE API SHALL return responses that validate against the response schema.
6. WHERE the optional HTTP wrapper is started, THE API SHALL serve the three functions over the Python standard library `http.server` bound to 127.0.0.1 and document that the wrapper has no authentication.
7. THE API SHALL provide no challenge-submission interface and no frontend.

### Requirement 13: Labeled end-to-end demo

**User Story:** As a hackathon presenter, I want one command that walks from human reasoning to a climate estimate, so that judges see the full chain with honest labels.

#### Acceptance Criteria

1. WHEN `python -m climate_intelligence.demo` is run, THE Demo SHALL print, in order, valuation of sample submissions, quality-gated selection, compute-efficiency comparison, and climate estimate.
2. THE Demo SHALL prefix every printed number with its label MEASURED, ESTIMATED, HYPOTHETICAL, or SYNTHETIC.
3. THE Demo SHALL include a fluent incorrect contrarian submission and show that submission excluded from training selection.
4. WHERE the `--synthetic` flag is given or the embeddings cache is missing, THE Demo SHALL run in synthetic mode and print a banner stating that results are synthetic and not evidence.
5. WHEN the Lab shows no significant Arm difference, THE Demo SHALL print that null result.
6. THE Demo SHALL complete without network access.

### Requirement 14: Tests, docs and repository safety

**User Story:** As a maintainer, I want tests and docs that run offline and leave existing work intact, so that the package can be trusted and extended.

#### Acceptance Criteria

1. THE Package SHALL include pytest tests under `tests/climate_intelligence/` that run without network access.
2. THE Package tests SHALL cover the incorrect-but-novel, correct-but-conventional, and unverified valuation cases.
3. THE Package tests SHALL cover the energy formulas, status propagation, input validation, leakage detection, token-budget matching, schema validation, and API Error_Objects.
4. THE Package SHALL include a README with setup, CLI usage, API examples, JSON Schema locations, and the status label definitions.
5. WHEN the Package tests run, THE existing tests in `tests/test_scoring.py` SHALL continue to pass.
6. THE Package SHALL leave existing files, `engine/`, `unbound/`, and `data/experiment/results.json` unmodified.
