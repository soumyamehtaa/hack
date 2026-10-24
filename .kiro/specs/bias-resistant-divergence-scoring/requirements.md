# Requirements Document

## Introduction

DIVERGE looks for unconventional human reasoning that could make AI training more data-efficient. The current scorer (`diverge/scoring.py`, version `diverge-score-0.1`) measures originality (semantic divergence, conceptual novelty, information gain) and quality (reasoning, evidence). It gates originality by reasoning quality. It has no notion of correctness, so a fluent but wrong contrarian answer can still score well. A correct answer that agrees with the AI also scores low, because most of the score weight goes to distance from the AI Baselines.

This feature extends the existing scorer. It does not replace it. It adds four separate signals (Semantic_Novelty, Reasoning_Quality, Correctness_Status, Marginal_Value), combines them into an Intelligence_Value_Score with a hard cap for incorrect answers and a floor for correct conventional answers, and reports how the score was built, how uncertain it is, and whether it was verified. AI Baselines are used only as a reference point for novelty, never as a source of truth.

Scope is one day of work. Reuse `models.py` (MiniLM embeddings, DeBERTa NLI, Qwen2.5-0.5B perplexity), `baselines.py`, `scoring.py`, `store.py` and `questions.py`. New pieces: a small human-authored reference file for verifiable questions, a correctness checker, the combination and uncertainty logic, a versioned evaluation configuration, and a bias and edge-case test suite. The existing spec `diverge-intelligence-platform` stays unchanged. Its Divergence_Score (D) and Verification_Status terms keep their meaning. This feature's Intelligence_Value_Score is the correctness-aware successor to that spec's Intelligence_Value (V).

Amendment (hackathon day): the score combination in `diverge/value.py` (`ivs-0.1-experimental`) exists but is not wired into `app.py`. Requirements 11 to 14 wire IVS into the Arena, split the Arena display into an originality-only Divergence_Score_D and a separate IVS, make training-data eligibility depend on IVS and Correctness_Status instead of the legacy composite, and add objectively verifiable questions (q11 onward) whose Reference_Keys carry an exact Expected_Answer. The seven App tabs (Arena, Consensus Breaker, Leaderboard, Training Data, Compute Experiment, Climate Impact, Method), the 11 existing records in `data/submissions.jsonl`, the 3 export formats and the single Streamlit server are preserved. Divergence_Score_D is a distance measure, not a quality measure: it is deliberately ungated for display, so the Quality_Gate continues to apply inside IVS and IVS alone (with Correctness_Status) decides training eligibility.

Amendment (product redesign, supersedes every "seven tabs preserved" statement above and in Requirement 12): the App moves from a seven-tab dashboard to four primary pages, HOME, CHALLENGES, MY IMPACT and ABOUT (Requirements 15 to 20). In UI copy the Arena becomes the Challenge_Workspace and IVS is shown as "Estimated Learning Value". Earlier criteria that name the Arena apply to the Challenge_Workspace; criteria that name the Leaderboard apply to the My_Impact_Page leaderboard; criteria that name the Training Data view apply to the Export_Panel. Consensus Breaker and JSONL export stay reachable outside primary navigation; the compute experiment and climate calculator move to the About_Page. The backend modules `scoring.py`, `value.py`, `store.py`, `baselines.py`, `experiment.py` and `seed.py`, and the files `data/submissions.jsonl`, `data/baselines.json` and `data/experiment/results.json`, are preserved and reused. The App still runs as one local Streamlit server.

## Glossary

- **Scorer**: The scoring module in `diverge/scoring.py`, extended by this feature.
- **Correctness_Checker**: New component that decides Correctness_Status from a Reference_Key.
- **Store**: The JSONL submission store in `diverge/store.py`.
- **App**: The DIVERGE Streamlit application in `app.py`.
- **Submission**: A user's answer text plus optional reasoning text for one question.
- **Baseline**: One of the cached AI answers for a question in `data/baselines.json`.
- **Consensus_Assumption**: A claim that recurs across at least 2 Baselines, as returned by `consensus_assumptions()`.
- **Prior_Pool**: The Baselines plus all earlier Submissions for the same question that were used when a Submission was scored.
- **Reference_Key**: A human-authored entry for one question, stored in `data/references.json`. It holds one or more Accepted_Claims, zero or more Rejected_Claims, and a source citation for each claim. Reference_Keys are never generated from Baselines or by any LLM.
- **Accepted_Claim**: A statement in a Reference_Key that a correct answer must entail.
- **Rejected_Claim**: A statement in a Reference_Key that is a known misconception. An answer that asserts it is incorrect.
- **Verifiable_Question**: A question that has a Reference_Key.
- **Correctness_Status**: One of `verified_correct`, `verified_incorrect`, `unverified`.
- **Correctness_Credit**: The 0–100 value used in Merit_Score for Correctness_Status: 100 for `verified_correct`, 50 for `unverified`, 0 for `verified_incorrect`.
- **verification evidence**: The record of the (claim, Submission sentence) pair, NLI relation, probability and source citation that explains a Correctness_Status, or a note such as "no reference key" or "verification unavailable".
- **load-warnings list**: The list of warnings the Correctness_Checker collects while loading `data/references.json`, readable from the Correctness_Checker after initialization.
- **Semantic_Novelty**: 0–100 value combining the existing semantic divergence and conceptual novelty components.
- **Reasoning_Quality**: 0–100 value equal to the existing `reasoning_quality` output × 100.
- **Evidence_Quality**: 0–100 value equal to the existing `evidence_quality` output × 100.
- **Marginal_Value**: 0–100 estimate of how much new learning signal a Submission adds beyond the Prior_Pool, from the existing `information_gain` output. It is a proxy, not a measurement.
- **Quality_Gate**: The existing factor clip((Reasoning_Quality/100 − 0.25) / 0.35, 0, 1) that scales originality credit.
- **Novelty_Credit**: The part of the Intelligence_Value_Score earned from Semantic_Novelty and Marginal_Value after the Quality_Gate is applied.
- **Merit_Score**: The part of the Intelligence_Value_Score earned from Reasoning_Quality, Evidence_Quality and Correctness_Status. It has no novelty input.
- **Intelligence_Value_Score (IVS)**: 0–100 value combining Merit_Score and Novelty_Credit, then applying the Incorrect_Cap and Conventional_Floor rules.
- **Incorrect_Cap**: The maximum IVS for a `verified_incorrect` Submission. Default 25.
- **Conventional_Floor**: The minimum IVS for a `verified_correct` Submission whose Reasoning_Quality is at least Floor_Quality_Threshold. Default 55.
- **Floor_Quality_Threshold**: The Reasoning_Quality needed for the Conventional_Floor to apply. Default 50.
- **Entailment_Threshold**: The NLI probability needed to count a claim as entailed or contradicted. Default 0.7.
- **Evaluation_Config**: A versioned set of every weight, threshold, cap, floor, model identifier, per-device numeric precision and random seed that affects scoring.
- **execution environment**: The compute device (`cpu` or `mps`), the numeric precision of each local model, and the installed versions of torch, transformers and sentence-transformers used when a Submission is scored.
- **Config_Hash**: SHA-256 hash of the canonical JSON serialization of an Evaluation_Config.
- **Score_Interval**: The pair (IVS_Low, IVS_High) that bounds the IVS across the uncertainty sources defined in Requirement 7.
- **IVS_Low / IVS_High**: The lower and upper bounds of the Score_Interval.
- **Uncertainty_Level**: `low` when IVS_High − IVS_Low < 10, `medium` when the width is 10 to 25, `high` when the width is > 25, or `high` when an uncertainty reason applies (Requirement 7).
- **uncertainty reasons**: The list of reasons ("borderline verification", "insufficient baselines") that forced Uncertainty_Level to `high` for a Submission; empty when none apply.
- **Score_Breakdown**: The per-component record of raw value, weight, credited contribution, and any cap or floor adjustment.
- **Bias_Test_Suite**: The automated pytest and Hypothesis tests defined in Requirement 10.
- **Divergence_Score_D**: 0–100 originality-only score 100 × (0.50 × semantic_divergence + 0.30 × claim_novelty + 0.20 × assumption_challenge), using the existing Scorer outputs `semantic_divergence`, `details.claim_novelty` and `details.assumption_challenge` (each 0–1), with no quality, evidence or correctness input and no Quality_Gate.
- **Legacy_Composite**: The existing `score` field produced by the `diverge-score-0.1` formula, kept for backward compatibility.
- **Verifiable_Flag**: A boolean `verifiable` field on a question entry in `diverge/questions.py`; absent means false.
- **Expected_Answer**: Optional part of a Reference_Key holding a `kind` (`numeric` or `string`), the expected value (a number with an absolute tolerance ≥ 0 for `numeric`; 1 to 10 accepted aliases for `string`), zero or more known wrong values or aliases, and a source citation.
- **Extracted_Answer**: For `numeric` Expected_Answers, the first numeric token in the Submission's answer text after removing currency symbols and thousands separators, where `a/b` is read as a fraction and a token ending in `%` is compared both as written and divided by 100; for `string` Expected_Answers, the answer text lowercased with punctuation removed.
- **IVS_Threshold**: Configurable minimum IVS for a record to be Training_Grade, held in the Evaluation_Config. Default 45.
- **Training_Grade**: A record whose IVS ≥ IVS_Threshold, whose Correctness_Status is not `verified_incorrect`, and whose Training_Consent is true. Demo-seed records are exempt from the consent condition. Records without a Training_Consent field are treated as consent false.
- **Training_Consent**: A boolean `training_consent` field stored on a Submission, set from the consent checkbox in the Challenge_Workspace.
- **Display_Name**: The contributor name entered in the Challenge_Workspace or My_Impact_Page, 1 to 40 characters after trimming, containing no "@"; stored in the existing `user` field.
- **Home_Page**, **Challenges_Page**, **My_Impact_Page**, **About_Page**: The four primary pages of the App.
- **Challenge_Workspace**: The two-column view on the Challenges_Page for one selected question; the successor of the Arena.
- **Estimated_Learning_Value**: The user-facing name of the IVS.
- **Verification_Type**: "Objective answer" for a question with Verifiable_Flag true and a valid Reference_Key holding an Expected_Answer; "Open-ended" otherwise.
- **Solved**: A question for which the current Display_Name has at least 1 stored Submission.
- **Experiment_Results**: The stored output of `diverge/experiment.py` in `data/experiment/results.json`; the only source of measured experiment values.
- **Energy_Calculator**: The About_Page scenario tool computing Energy (kWh) = Average Power (kW) × Duration (hours), scenario duration = baseline duration × (1 − assumed compute reduction), and scenario savings (kWh) = baseline energy − scenario energy. All inputs are user-set assumptions.
- **Export_Panel**: The JSONL export view with the Requirement 8 criterion 3 and Requirement 13 filters, reachable from the My_Impact_Page or About_Page.

## Requirements

### Requirement 1: Separate evaluation dimensions

**User Story:** As a dataset curator, I want novelty, reasoning, correctness and marginal value measured separately, so that one signal cannot hide a weakness in another.

#### Acceptance Criteria

1. WHEN a Submission is scored, THE Scorer SHALL report Semantic_Novelty, Reasoning_Quality, Evidence_Quality and Marginal_Value as 4 separate numeric fields, each clipped to the range 0–100 and rounded to 1 decimal place, and Correctness_Status as a separate field whose value is exactly one of `verified_correct`, `verified_incorrect` or `unverified`.
2. THE Scorer SHALL compute Semantic_Novelty as 100 × (0.55 × semantic_divergence + 0.45 × conceptual_novelty), where semantic_divergence is the existing `semantic_divergence` output and conceptual_novelty is the existing 0.6 × claim novelty + 0.4 × assumption challenge value, both computed with the same Baselines and Prior_Pool as the existing `components` field and both taken before any rounding.
3. THE Scorer SHALL compute Reasoning_Quality, Evidence_Quality and Marginal_Value as 100 × the existing `reasoning_quality`, `evidence_quality` and `information_gain` outputs respectively, so that each reported value equals the matching value in the existing `components` field × 100 rounded to 1 decimal place.
4. THE Scorer SHALL compute Semantic_Novelty and Marginal_Value without Correctness_Status, Reference_Key content, Reasoning_Quality or Evidence_Quality as inputs, so that adding, changing or removing the question's Reference_Key (including a missing `data/references.json`) leaves both values identical for the same question, Submission text, Baselines and Prior_Pool.
5. THE Scorer SHALL compute Correctness_Status from the Submission text and the question's Reference_Key only, without Semantic_Novelty, Marginal_Value, Reasoning_Quality, Evidence_Quality, Prior_Pool or any similarity to Baselines as inputs.
6. THE Scorer SHALL continue to produce the existing `score`, `originality`, `quality`, `gate`, `components`, `credited`, `contributions`, `details`, `claims`, `assumptions` and `explanation` fields with values identical to those produced by `diverge-score-0.1` for the same question, Submission text, Baselines and Prior_Pool, and SHALL place all new fields, including the new explanation sentences of Requirement 6, outside these existing fields, so that existing App views and legacy records keep working.
7. IF the Correctness_Checker raises an error while scoring a Submission, THEN THE Scorer SHALL set Correctness_Status to `unverified`, still report Semantic_Novelty, Reasoning_Quality, Evidence_Quality, Marginal_Value and the existing fields of criterion 6, and record a warning indicating that correctness verification failed for that question id.

### Requirement 2: Reference keys for verifiable questions

**User Story:** As a researcher, I want correctness to come from human-authored references, so that the AI's opinion is never the yardstick.

#### Acceptance Criteria

1. WHEN the Correctness_Checker is initialized, THE Correctness_Checker SHALL load Reference_Keys from `data/references.json` once, keyed by any question id defined in `diverge/questions.py` (`q01` to `q10` and any added questions such as `q11` to `q14`), and SHALL use the loaded Reference_Keys for every Submission scored until the Correctness_Checker is initialized again.
2. THE Correctness_Checker SHALL accept a Reference_Key only when all of the following hold: the key has 1 to 10 Accepted_Claims and 0 to 10 Rejected_Claims; every claim text is 1 to 500 characters after trimming leading and trailing whitespace; every Accepted_Claim and Rejected_Claim has a source citation of 1 to 500 characters after trimming whitespace; and no claim text, compared case-insensitively after trimming whitespace, appears as both an Accepted_Claim and a Rejected_Claim of the same key; and, when an Expected_Answer is present, its `kind` is `numeric` or `string`, its value matches that kind, its citation is 1 to 500 characters after trimming, and no known wrong value equals the expected value (within tolerance for `numeric`, case-insensitively for `string`).
3. IF a Reference_Key is malformed (it is not a structured entry holding a list of Accepted_Claims, a list of Rejected_Claims and a citation per claim, or any claim or citation is not text) or fails any check in criterion 2, THEN THE Correctness_Checker SHALL treat that question as not a Verifiable_Question, SHALL keep accepting every other valid Reference_Key in the file, and SHALL add a warning naming the question id and the failed check to the load-warnings list.
4. IF `data/references.json` is missing or cannot be parsed as JSON, THEN THE Correctness_Checker SHALL treat every question as not a Verifiable_Question and add one warning naming the file and the failure reason to the load-warnings list, and THE Scorer SHALL keep scoring Submissions with Correctness_Status `unverified` and without raising an error.
5. IF `data/references.json` contains a Reference_Key whose question id is not one of the question ids in `diverge/questions.py`, THEN THE Correctness_Checker SHALL ignore that Reference_Key and add a warning naming the unknown id to the load-warnings list.
6. THE feature SHALL ship `data/references.json` with Reference_Keys for at least 3 of the questions `q01` to `q10` in `diverge/questions.py` (in addition to the Reference_Keys of Requirement 14), where each shipped Reference_Key passes every check in criterion 2, has at least 1 Accepted_Claim and at least 1 Rejected_Claim, and produces 0 load warnings when loaded.
7. THE shipped Reference_Key for at least 1 question SHALL include an Accepted_Claim for which at least 1 Consensus_Assumption of that question's cached Baselines (the `data/baselines.json` entry whose question text equals the text of that question id in `diverge/questions.py`) has a DeBERTa NLI contradiction probability ≥ Entailment_Threshold, with the Accepted_Claim as premise and the Consensus_Assumption as hypothesis, so that the case "AI consensus is wrong" is covered by real data.
8. WHEN the Bias_Test_Suite runs with local models installed, THE Bias_Test_Suite SHALL verify criterion 7 against the shipped `data/references.json` and the cached `data/baselines.json`, and SHALL fail if no shipped Accepted_Claim meets criterion 7.

### Requirement 3: Correctness verification

**User Story:** As a dataset curator, I want each Submission marked as verified correct, verified incorrect or unverified, so that I know which claims have been checked.

#### Acceptance Criteria

1. WHEN a Submission for a Verifiable_Question is scored, THE Correctness_Checker SHALL split the Submission's answer text and reasoning text (joined in that order) into sentences using the same sentence splitter the Scorer uses, and SHALL compute, for every pair of Submission sentence and Accepted_Claim or Rejected_Claim, the DeBERTa NLI (`cross-encoder/nli-deberta-v3-xsmall`) entailment and contradiction probabilities, each in the range 0.0–1.0, with the claim as premise and the sentence as hypothesis.
2. IF the Reference_Key has no Expected_Answer and at least 1 Submission sentence has entailment probability ≥ Entailment_Threshold for a Rejected_Claim, or contradiction probability ≥ Entailment_Threshold for an Accepted_Claim, THEN THE Correctness_Checker SHALL set Correctness_Status to `verified_incorrect`, regardless of how many Accepted_Claims are also entailed.
3. IF the Reference_Key has no Expected_Answer, criterion 2 does not apply, and at least 1 Accepted_Claim has entailment probability ≥ Entailment_Threshold with at least 1 Submission sentence, THEN THE Correctness_Checker SHALL set Correctness_Status to `verified_correct`.
4. IF a Submission is for a Verifiable_Question whose Reference_Key has no Expected_Answer and neither criterion 2 nor criterion 3 applies, or the Submission's answer and reasoning text together contain 0 sentences after whitespace trimming, THEN THE Correctness_Checker SHALL set Correctness_Status to `unverified`.
5. IF a Submission is for a question that is not a Verifiable_Question, THEN THE Correctness_Checker SHALL set Correctness_Status to `unverified` without running any NLI computation and SHALL record verification evidence noting "no reference key".
6. WHEN Correctness_Status is set to `verified_incorrect` or `verified_correct`, THE Correctness_Checker SHALL record as verification evidence the single (claim, sentence) pair with the highest qualifying probability among the pairs that met the deciding criterion, including the claim text, whether the claim is Accepted or Rejected, the Submission sentence text, the NLI relation (entailment or contradiction), the probability rounded to 3 decimal places, and the claim's source citation.
7. WHEN Correctness_Status is set to `unverified` for a Verifiable_Question, THE Correctness_Checker SHALL record as verification evidence the (claim, sentence) pair with the highest entailment or contradiction probability observed (rounded to 3 decimal places), or no pair when the Submission has 0 sentences.
8. IF the NLI model cannot be loaded or fails during inference, THEN THE Correctness_Checker SHALL set Correctness_Status to `unverified`, record verification evidence noting "verification unavailable", and THE Scorer SHALL continue to produce all other score fields for the Submission.
9. FOR ALL Submissions and Reference_Keys, replacing the Baselines with any other set of Baselines (including an empty set) SHALL leave Correctness_Status and verification evidence unchanged (AI consensus is not ground truth).

### Requirement 4: Intelligence Value Score with cap and floor

**User Story:** As an ML engineer, I want a single value score that rewards being right and useful, so that a wrong contrarian answer never outranks a right conventional one.

#### Acceptance Criteria

1. WHEN a Submission is scored, THE Scorer SHALL compute Merit_Score as w_reasoning × Reasoning_Quality + w_evidence × Evidence_Quality + w_correctness × Correctness_Credit, where w_reasoning, w_evidence and w_correctness are read from the Evaluation_Config, Reasoning_Quality and Evidence_Quality are the 0–100 values reported under Requirement 1, and Correctness_Credit is 100 for `verified_correct`, 50 for `unverified` and 0 for `verified_incorrect`, with no other input.
2. WHEN a Submission is scored, THE Scorer SHALL compute Novelty_Credit as Quality_Gate × (w_novelty × Semantic_Novelty + w_marginal × Marginal_Value), where w_novelty and w_marginal are read from the Evaluation_Config and Semantic_Novelty and Marginal_Value are the 0–100 values reported under Requirement 1.
3. WHEN a Submission is scored, THE Scorer SHALL compute IVS by summing Merit_Score and Novelty_Credit, then applying criterion 4 or criterion 5 (at most one applies, since they require different Correctness_Status values), then clipping the result to the range 0–100, then rounding the result to 1 decimal place, in that order.
4. IF Correctness_Status is `verified_incorrect`, THEN THE Scorer SHALL set IVS to min(IVS, Incorrect_Cap) before clipping and rounding.
5. WHEN a Submission with Correctness_Status `verified_correct` and Reasoning_Quality ≥ Floor_Quality_Threshold (comparing the 1-decimal Reasoning_Quality value) is scored, THE Scorer SHALL set IVS to max(IVS, Conventional_Floor) before clipping and rounding; a `verified_correct` Submission with Reasoning_Quality below Floor_Quality_Threshold SHALL receive no floor adjustment.
6. THE Evaluation_Config SHALL hold w_reasoning, w_evidence, w_correctness, w_novelty and w_marginal each in 0–1 with a sum of 1.0 (±0.001), Incorrect_Cap and Conventional_Floor each in 0–100 with Incorrect_Cap strictly below Conventional_Floor, and Floor_Quality_Threshold in 0–100.
7. FOR ALL pairs of Submissions A and B with identical Reasoning_Quality, Evidence_Quality, Correctness_Status and Evaluation_Config, where Semantic_Novelty_A ≥ Semantic_Novelty_B and Marginal_Value_A ≥ Marginal_Value_B, THE Scorer SHALL produce IVS_A ≥ IVS_B (novelty is a bonus, never a penalty).
8. FOR ALL Submissions with Correctness_Status `verified_incorrect`, THE Scorer SHALL produce IVS ≤ Incorrect_Cap for every combination of Semantic_Novelty, Marginal_Value, Reasoning_Quality and Evidence_Quality in 0–100.
9. FOR ALL Submissions with Correctness_Status `verified_correct` and Reasoning_Quality ≥ Floor_Quality_Threshold, THE Scorer SHALL produce IVS ≥ Conventional_Floor for every combination of Semantic_Novelty, Marginal_Value and Evidence_Quality in 0–100, including when Semantic_Novelty, Marginal_Value and Evidence_Quality are all 0.
10. FOR ALL triples of Submissions with identical Semantic_Novelty, Reasoning_Quality, Evidence_Quality, Marginal_Value and Evaluation_Config that differ only in Correctness_Status, THE Scorer SHALL produce IVS(`verified_correct`) ≥ IVS(`unverified`) ≥ IVS(`verified_incorrect`).
11. IF the Evaluation_Config violates any constraint in criterion 6, THEN THE Scorer SHALL refuse to load that Evaluation_Config, score no Submission with it, and raise an error naming the violated constraint and the offending value.
12. IF any of Semantic_Novelty, Reasoning_Quality, Evidence_Quality or Marginal_Value passed to the score-combination logic is non-numeric, non-finite, or outside 0–100, or Correctness_Status is not one of `verified_correct`, `verified_incorrect` or `unverified`, THEN THE Scorer SHALL raise an error naming the invalid input and SHALL NOT produce an IVS for that Submission.

### Requirement 5: Freedom from AI-consensus bias

**User Story:** As a researcher, I want agreement or disagreement with the AI to be neutral for correctness, so that valid contrarian reasoning and valid conventional reasoning are both treated fairly.

#### Acceptance Criteria

1. THE Scorer SHALL compute Merit_Score without any similarity to Baselines, agreement with Consensus_Assumptions, or contradiction of Consensus_Assumptions as inputs, so that replacing the Baselines of a question with any other set of 1 or more Baselines leaves Merit_Score unchanged within ±0.1 for the same question text, Submission text and Evaluation_Config.
2. WHEN a Submission contradicts 1 or more Consensus_Assumptions (DeBERTa NLI contradiction probability ≥ Entailment_Threshold, with the Consensus_Assumption as premise and a Submission sentence as hypothesis) and the Correctness_Checker sets Correctness_Status to `verified_correct`, THE Scorer SHALL report Correctness_Status `verified_correct` and SHALL apply a Correctness_Credit of 100 in Merit_Score, identical to a `verified_correct` Submission that contradicts no Consensus_Assumption.
3. WHEN a Submission's maximum cosine similarity to any Baseline, computed with the existing MiniLM embedding model as in `semantic_divergence`, is ≥ 0.85, Correctness_Status is `verified_correct`, and Reasoning_Quality ≥ Floor_Quality_Threshold, THE Scorer SHALL produce IVS ≥ Conventional_Floor and SHALL compute Merit_Score with the same weights and Correctness_Credit of 100 as for any other `verified_correct` Submission.
4. THE Scorer SHALL compute every score field (Semantic_Novelty, Reasoning_Quality, Evidence_Quality, Marginal_Value, Correctness_Status, Merit_Score, Novelty_Credit, IVS, Score_Interval, Uncertainty_Level, Score_Breakdown, Divergence_Score_D, and the legacy `score`, `originality`, `quality`, `gate` and `components` fields) from the question text, Submission text, Baselines, Prior_Pool and Evaluation_Config only.
5. WHEN two Submissions have identical question text, Submission text, Baselines, Prior_Pool and Evaluation_Config but differ in author name, submission timestamp, demo-seed flag, submission identifier or any other stored metadata, THE Scorer SHALL return equal values for every score field listed in criterion 4, excluding only the run-timing metadata (scoring duration and scored-at time).
6. IF a Submission entails 1 or more Consensus_Assumptions (DeBERTa NLI entailment probability ≥ Entailment_Threshold) and the Correctness_Checker sets Correctness_Status to `verified_incorrect`, THEN THE Scorer SHALL report Correctness_Status `verified_incorrect` and SHALL produce IVS ≤ Incorrect_Cap, so that agreement with the AI does not raise the score of an incorrect answer.
7. IF a Submission for a question with no Reference_Key entails or contradicts 1 or more Consensus_Assumptions with NLI probability ≥ Entailment_Threshold, THEN THE Correctness_Checker SHALL set Correctness_Status to `unverified`, so that neither agreement nor disagreement with the AI produces a verified Correctness_Status.

### Requirement 6: Explainability

**User Story:** As a contributor, I want to see exactly why my answer got its score, so that I can trust or dispute it.

#### Acceptance Criteria

1. WHEN a Submission is scored, THE Scorer SHALL produce a Score_Breakdown with one entry per input component of Merit_Score (Reasoning_Quality, Evidence_Quality, Correctness_Credit) and of Novelty_Credit (Semantic_Novelty, Marginal_Value), each entry giving the component's raw value (0–100, 1 decimal place), its weight from the Evaluation_Config, the Quality_Gate factor (3 decimal places, for Novelty_Credit entries only) and its credited contribution in IVS points (2 decimal places).
2. THE Scorer SHALL ensure the credited contributions in the Score_Breakdown sum to the IVS before the Incorrect_Cap, Conventional_Floor and 0–100 clip are applied, within ±0.1, and SHALL record that pre-rule IVS in the Score_Breakdown.
3. WHEN the Incorrect_Cap or Conventional_Floor changes the IVS by more than 0, THE Scorer SHALL record in the Score_Breakdown the name of the rule that applied, the rule's threshold value from the Evaluation_Config, the IVS before the rule, and the IVS after the rule.
4. WHEN neither the Incorrect_Cap nor the Conventional_Floor changes the IVS, THE Scorer SHALL record in the Score_Breakdown that no cap or floor was applied, with the IVS before and after equal.
5. WHEN a Submission is scored, THE Scorer SHALL produce exactly one plain-language sentence for each of Semantic_Novelty, Reasoning_Quality, Correctness_Status and Marginal_Value, each sentence stating the dimension's value (1 decimal place) and the numeric value (at least 2 decimal places) of each underlying measured input from which that value was computed (semantic divergence and conceptual novelty for Semantic_Novelty; the existing reasoning-quality measure for Reasoning_Quality; information gain and Prior_Pool size for Marginal_Value).
6. WHEN the Quality_Gate factor is below 1.0, THE Scorer SHALL add one sentence stating the Quality_Gate factor, the Reasoning_Quality that produced it, and that Novelty_Credit was scaled by that factor.
7. WHEN Correctness_Status is `verified_correct` or `verified_incorrect`, THE Scorer SHALL include in the Correctness_Status sentence the deciding claim text, whether it is an Accepted_Claim or Rejected_Claim, the deciding Submission sentence, its source citation, and the NLI probability (2 decimal places) from the verification evidence.
8. WHEN Correctness_Status is `unverified`, THE Scorer SHALL state in the Correctness_Status sentence either that the question has no valid Reference_Key, or that no claim reached the Entailment_Threshold together with the highest NLI probability observed (2 decimal places) and the Entailment_Threshold value.
9. WHEN a scored Submission is displayed, THE App SHALL show, in the same view as the IVS, every Score_Breakdown entry, any cap or floor record, all explanation sentences, and the text label "estimated proxy, not measured" adjacent to the Marginal_Value value.
10. IF a displayed record has no Score_Breakdown, THEN THE App SHALL show a message indicating that no breakdown is available because the record has not been rescored, and SHALL NOT display a reconstructed or estimated breakdown.

### Requirement 7: Visible uncertainty

**User Story:** As a dataset curator, I want to see how confident the evaluation is, so that I do not over-trust a single number.

#### Acceptance Criteria

1. WHEN a Submission is scored with N ≥ 2 Baselines, THE Scorer SHALL compute the IVS once with all N Baselines and N more times with exactly one Baseline left out, where each leave-one-out run removes that Baseline from the Baselines, from the Prior_Pool and from the Consensus_Assumption computation and keeps Correctness_Status and the Evaluation_Config unchanged, and SHALL set IVS_Low and IVS_High to the minimum and maximum of those N + 1 IVS values, each rounded to 1 decimal place.
2. WHILE Correctness_Status is `unverified`, THE Scorer SHALL compute two extra IVS values with all available Baselines, one with Correctness_Status replaced by `verified_incorrect` (Incorrect_Cap applied) and one with Correctness_Status replaced by `verified_correct` (Conventional_Floor rule applied), SHALL set IVS_Low to the minimum and IVS_High to the maximum of the criterion 1 bounds and these two values, and SHALL keep the stored Correctness_Status `unverified`.
3. WHEN the Score_Interval is computed and neither criterion 5 nor criterion 7 applies, THE Scorer SHALL set Uncertainty_Level from the width W = IVS_High − IVS_Low, rounded to 1 decimal place: `low` if W < 10.0, `medium` if 10.0 ≤ W ≤ 25.0, and `high` if W > 25.0.
4. THE Scorer SHALL guarantee IVS_Low ≤ IVS ≤ IVS_High, with all three values in 0–100 and rounded to 1 decimal place, for every scored Submission.
5. IF the absolute difference between Entailment_Threshold and the NLI probability recorded as verification evidence for a `verified_correct` or `verified_incorrect` Submission is ≤ 0.1, or a Submission for a Verifiable_Question is `unverified` and its highest entailment or contradiction probability across all sentence–claim pairs is ≥ Entailment_Threshold − 0.1, THEN THE Scorer SHALL set Uncertainty_Level to `high` regardless of interval width and add the reason "borderline verification" to the Submission's uncertainty reasons.
6. WHEN a scored Submission is displayed, THE App SHALL show IVS, IVS_Low, IVS_High (each to 1 decimal place) and Uncertainty_Level in the same view, plus every recorded uncertainty reason whenever Uncertainty_Level is `high`.
7. IF fewer than 2 Baselines are available (0 or 1), THEN THE Scorer SHALL skip leave-one-out, set IVS_Low and IVS_High to the IVS (before any criterion 2 widening), set Uncertainty_Level to `high` regardless of interval width, and add the reason "insufficient baselines" to the Submission's uncertainty reasons.
8. WHEN a Submission is scored, THE Scorer SHALL record its uncertainty reasons as a list that is empty when neither criterion 5 nor criterion 7 applies and holds both "borderline verification" and "insufficient baselines" when both apply.
9. IF a displayed record has no Score_Interval (a legacy record that has not been rescored), THEN THE App SHALL show the IVS or legacy score without IVS_Low, IVS_High or Uncertainty_Level and SHALL show the label "not rescored" in their place.

### Requirement 8: Verified versus unverified submissions

**User Story:** As an ML engineer, I want verified and unverified submissions kept apart, so that I can choose how much to trust each in training data.

#### Acceptance Criteria

1. WHEN a scored Submission is displayed in the App's Arena result view, Leaderboard submission table or Training Data table, THE App SHALL show exactly one Correctness_Status badge per Submission, using the text label "Verified correct" for `verified_correct`, "Verified incorrect" for `verified_incorrect` and "Unverified" for `unverified`, with the label text alone being enough to tell the three states apart (no reliance on color only).
2. WHEN a Submission is accepted for storage, THE Store SHALL write a single JSONL record that contains Correctness_Status (one of `verified_correct`, `verified_incorrect`, `unverified`), verification evidence (deciding claim, deciding Submission sentence, NLI probability and source citation, or an empty value when Correctness_Status is `unverified` with no deciding claim), IVS, Score_Interval (IVS_Low and IVS_High), Uncertainty_Level, Score_Breakdown and Config_Hash, in addition to the fields stored today.
3. WHEN the user opens the Training Data view, THE App SHALL show 3 independent include/exclude toggles, one per Correctness_Status value, with `verified_correct` and `unverified` set to include and `verified_incorrect` set to exclude by default, and SHALL show only records whose Correctness_Status is included AND that pass the minimum IVS (default IVS_Threshold), minimum quality gate and demo-seed filters.
4. IF all Correctness_Status toggles are set to exclude, or no record passes the combined filters, THEN THE App SHALL show 0 selected records and disable the export button.
5. WHEN records are exported in any of the 3 export formats ("DIVERGE record (full)", "chat (fine-tuning)", "preference pair (DPO)"), THE Store SHALL include Correctness_Status, verification evidence, IVS, IVS_Low, IVS_High and Uncertainty_Level in every exported record (in the record's metadata for the chat and preference-pair formats), and SHALL export only the records the user selected.
6. WHEN a legacy record without a Correctness_Status field is loaded, THE Store SHALL treat the record's Correctness_Status as `unverified`, report its IVS, Score_Interval and verification evidence as empty values, and leave the stored file content unchanged, and THE App SHALL show the label "not rescored" next to the "Unverified" badge until the record is rescored.
7. WHEN a legacy record is exported before rescoring, THE Store SHALL export Correctness_Status as `unverified` and include a "not rescored" flag set to true in that exported record.
8. IF a loaded record has a Correctness_Status value other than `verified_correct`, `verified_incorrect` or `unverified`, THEN THE Store SHALL treat the record as `unverified` and leave the stored record unchanged, and THE App SHALL show the label "invalid status" next to its badge.
9. IF the Store fails to write an accepted Submission, THEN THE App SHALL show an error message indicating the Submission was not saved and keep the user's entered answer and reasoning text in the form, and THE Store SHALL leave no partial record in the submissions file.
10. FOR ALL records written by the Store then exported in any of the 3 export formats, parsing each exported JSONL line SHALL yield Correctness_Status, verification evidence, IVS, IVS_Low and IVS_High equal to the stored values, with numeric values equal after rounding to 1 decimal place (round-trip property).

### Requirement 9: Reproducibility under fixed configuration

**User Story:** As a researcher, I want identical inputs and configuration to give identical results, so that scores can be audited and compared.

#### Acceptance Criteria

1. THE Scorer SHALL read every weight, threshold, cap, floor, model identifier, per-device numeric precision (float16 on MPS, float32 on CPU) and seed from a single Evaluation_Config that carries a non-empty version string of at most 64 characters, and SHALL use no scoring constant defined outside the Evaluation_Config.
2. WHEN a Submission is scored, THE Scorer SHALL record in the result metadata the Evaluation_Config version, the Config_Hash, the SHA-256 hash of each Prior_Pool text's UTF-8 bytes (Baselines first in cached order, then earlier Submissions in ascending submission-time order), and the execution environment.
3. WHEN the same question, Submission text, Baselines, Prior_Pool and Evaluation_Config are scored twice with the same recorded execution environment, whether in one process or in two separate process runs, THE Scorer SHALL return identical values for every score field, Correctness_Status and Score_Interval after rounding to 1 decimal place, excluding only the scoring-duration and scored-at timestamp metadata fields.
4. WHEN a stored record is rescored with its stored Baselines, stored Prior_Pool, an Evaluation_Config matching the stored Config_Hash, and an execution environment matching the stored execution environment, THE Scorer SHALL reproduce the stored IVS and both Score_Interval bounds within ±0.1 and the stored Correctness_Status exactly.
5. IF the Evaluation_Config used for rescoring has a different Config_Hash than the stored record, THEN THE Scorer SHALL flag the result as "config changed", report both Config_Hashes, keep the stored record's values unchanged, and return the new values as a separate result.
6. THE Scorer SHALL run all local model inference that affects scoring (MiniLM embeddings, DeBERTa NLI, Qwen2.5-0.5B perplexity) in inference mode with dropout disabled and no sampling, with the Python, NumPy and torch random generators seeded from the Evaluation_Config seed before each Submission is scored, and with each model loaded at the numeric precision the Evaluation_Config sets for the active compute device; Baseline generation is not part of scoring.
7. FOR ALL Evaluation_Configs, serializing to canonical JSON (keys sorted, UTF-8, no whitespace between tokens) and parsing back SHALL produce an equal Evaluation_Config with an equal Config_Hash, and changing any single value SHALL produce a different Config_Hash (round-trip property).
8. IF the Evaluation_Config is missing a required value, has a value outside its documented range, or sets Incorrect_Cap greater than or equal to Conventional_Floor, THEN THE Scorer SHALL refuse to score, return an error naming each invalid field, and leave the Store unchanged.
9. IF the execution environment used for rescoring differs from the stored execution environment in compute device, numeric precision or any recorded library version, THEN THE Scorer SHALL flag the result as "environment changed", report both environment descriptions and the difference in IVS, Score_Interval bounds and Correctness_Status, and skip the ±0.1 tolerance check of criterion 4.
10. IF a stored record being rescored lacks a Config_Hash, stored Baselines or Prior_Pool hashes, or any available Prior_Pool text hashes to a value different from the stored hash at the same position, THEN THE Scorer SHALL flag the result as "not reproducible", name each missing or mismatched item, and keep the stored record's values unchanged.

### Requirement 10: Bias and edge-case test suite

**User Story:** As a developer, I want automated tests that check the scoring guarantees, so that later changes to weights or models cannot silently break them.

#### Acceptance Criteria

1. THE Bias_Test_Suite SHALL run with a single `pytest` invocation and use Hypothesis for property-based tests, with at least 100 generated examples per property and a fixed Hypothesis seed (derandomized), so that two runs on the same code generate the same examples.
2. THE Bias_Test_Suite SHALL test the score-combination logic (Merit_Score, Novelty_Credit, Incorrect_Cap, Conventional_Floor, Score_Interval) as a pure function of component values, drawing Semantic_Novelty, Reasoning_Quality, Evidence_Quality and Marginal_Value from 0–100 inclusive and Correctness_Status from all 3 values, without loading any model, and the pure-function tests SHALL complete in 60 seconds or less on a developer laptop.
3. THE Bias_Test_Suite SHALL include property tests for Requirement 4 criteria 7, 8, 9 and 10, Requirement 7 criterion 4, and Requirement 9 criterion 7, each run against the default Evaluation_Config.
4. THE Bias_Test_Suite SHALL include a property test that, for random Submission metadata (author name of 0–100 Unicode characters, timestamp between 2000-01-01 and 2100-12-31, demo-seed flag true or false) and one fixed Submission text, asserts that every score field is equal across all generated metadata (Requirement 5 criteria 4 and 5), with model outputs replaced by fixed stub values so that no model is loaded.
5. THE Bias_Test_Suite SHALL include fixture tests using the real local models, with hand-written fixture Submissions for a question that has a shipped Reference_Key, for these cases: (a) a contrarian answer asserting a Rejected_Claim with Reasoning_Quality ≥ Floor_Quality_Threshold, asserting Correctness_Status `verified_incorrect` and IVS ≤ Incorrect_Cap; (b) an answer with cosine similarity ≥ 0.85 to one Baseline that entails an Accepted_Claim with Reasoning_Quality ≥ Floor_Quality_Threshold, asserting Correctness_Status `verified_correct` and IVS ≥ Conventional_Floor; (c) an answer that contradicts at least 1 Consensus_Assumption and entails an Accepted_Claim, asserting Correctness_Status `verified_correct`; and (d) an answer of 30 words sampled with a fixed seed from a word list without punctuation, asserting Quality_Gate 0 and Novelty_Credit 0.
6. THE Bias_Test_Suite SHALL include edge-case tests that assert scoring completes without raising an exception and with every 0–100 score field inside 0–100 for: an empty reasoning field; a single-sentence answer; a question without a Reference_Key (asserting Correctness_Status `unverified`); a missing `data/references.json` (asserting Correctness_Status `unverified` for every question); a malformed Reference_Key (asserting Correctness_Status `unverified` and a recorded warning naming the question id); and fewer than 2 Baselines (asserting Uncertainty_Level `high` with reason "insufficient baselines").
7. THE Bias_Test_Suite SHALL include a determinism test that scores the same fixture Submission twice under one Evaluation_Config and asserts that every score field, Correctness_Status and Score_Interval are equal after rounding to 1 decimal place (Requirement 9 criterion 3).
8. WHERE a local model (MiniLM embeddings, DeBERTa NLI or Qwen2.5-0.5B) cannot be loaded from the local cache, THE Bias_Test_Suite SHALL skip the model-backed fixture, edge-case and determinism tests without attempting a network download, report a skip reason naming the missing model, and still run the pure-function property tests.
9. IF any property, fixture or edge-case assertion fails, THEN THE Bias_Test_Suite SHALL exit with a non-zero status and report the failing test name together with the input component values or fixture Submission that caused the failure.
10. THE Bias_Test_Suite SHALL include a property test asserting that a record with Correctness_Status `verified_incorrect` is not Training_Grade for every Divergence_Score_D in 0–100, including Divergence_Score_D = 100.
11. THE Bias_Test_Suite SHALL include a property test asserting that Divergence_Score_D is unchanged when Reasoning_Quality, Evidence_Quality, Quality_Gate or Correctness_Status change while semantic_divergence, claim_novelty and assumption_challenge stay fixed.
12. THE Bias_Test_Suite SHALL include model-free tests for every shipped Expected_Answer of Requirement 14, asserting `verified_correct` for an answer stating the expected value, `verified_incorrect` for an answer stating each listed known wrong value, and `unverified` for an answer with no Extracted_Answer.
13. THE Bias_Test_Suite SHALL include tests asserting that legacy records (records without Correctness_Status, IVS or Divergence_Score_D fields, such as those in `data/submissions.jsonl` before this feature) load, render in the Leaderboard and Training Data tables, and export in all 3 export formats without raising an exception, with the source file unchanged afterwards.
14. THE Bias_Test_Suite SHALL include a test asserting that the `demo:noise` and `demo:offtopic` seed Submissions are not Training_Grade.

### Requirement 11: Originality-only Divergence Score in the Arena

**User Story:** As a contributor, I want to see how different my answer is separately from how valuable it is, so that originality and quality are never blended into one opaque number.

#### Acceptance Criteria

1. WHEN a Submission is scored, THE Scorer SHALL compute Divergence_Score_D as 100 × (0.50 × semantic_divergence + 0.30 × claim_novelty + 0.20 × assumption_challenge), using the ungated component values (each in the range 0.0–1.0) produced in the same scoring run, clip the result to 0.0–100.0, round it to 1 decimal place, and store it as a new field outside the existing fields listed in Requirement 1 criterion 6.
2. THE Scorer SHALL compute Divergence_Score_D without Reasoning_Quality, Evidence_Quality, Quality_Gate, Correctness_Status or Reference_Key content as inputs, such that changing any of these inputs while holding semantic_divergence, claim_novelty and assumption_challenge fixed leaves Divergence_Score_D unchanged.
3. WHEN Divergence_Score_D is introduced, THE Scorer SHALL change its VERSION string to a value different from `diverge-score-0.1`, record that VERSION in each newly scored Submission, and document the Divergence_Score_D formula and its experimental, not-validated status in the `diverge/scoring.py` module docstring, while leaving the Legacy_Composite values unchanged per Requirement 1 criterion 6.
4. WHEN a scored Submission is displayed in the Arena, THE App SHALL show Divergence_Score_D labeled "Divergence Score (originality only, not a quality measure)" and SHALL show the IVS in a separate visualization in the same result view, with neither value combined into the other.
5. WHERE the App displays the Legacy_Composite, THE App SHALL label the value "legacy composite".
6. THE Store SHALL keep storing and exporting the Legacy_Composite in its existing `score` and `divergence_score` positions so that existing export consumers keep working, and SHALL never write Divergence_Score_D into those positions.
7. IF a stored Submission has no Divergence_Score_D field (legacy record), THEN THE App and THE Store SHALL load, render and export that record without error, SHALL display Divergence_Score_D as "not computed" rather than deriving it from the Legacy_Composite, and SHALL leave the stored record unmodified.
8. IF any of semantic_divergence, claim_novelty or assumption_challenge is unavailable for a Submission being scored, THEN THE Scorer SHALL leave Divergence_Score_D empty instead of substituting a default value, and THE App SHALL display it as "not computed".

### Requirement 12: IVS integration in the Arena

**User Story:** As a contributor, I want my Arena result to show a correctness-aware value score, so that I see whether my answer is useful as well as different.

#### Acceptance Criteria

1. WHEN a Submission is scored in the Arena, THE App SHALL obtain Correctness_Status from the Correctness_Checker and compute IVS with `diverge/value.py` from the Requirement 1 dimensions (Semantic_Novelty, Reasoning_Quality, Evidence_Quality, Marginal_Value) and that Correctness_Status, and SHALL display IVS (0–100, 1 decimal place) and Divergence_Score_D as two separately labeled values in the same result view.
2. WHEN a Submission scored in the Arena is accepted for storage, THE Store SHALL add Divergence_Score_D, IVS, the IVS version string and the Requirement 8 criterion 2 fields to the record as new fields, leaving every existing field and its value unchanged, so that the stored record still loads in the Leaderboard and Training Data views and exports in all 3 export formats.
3. WHILE Correctness_Status is `unverified`, THE App SHALL display the "Unverified" badge next to the IVS together with the text "correctness not checked against a human reference".
4. WHILE Correctness_Status is `verified_correct` or `verified_incorrect`, THE App SHALL display a badge next to the IVS naming that status, and WHILE Correctness_Status is `verified_incorrect`, THE App SHALL show that the IVS is limited to at most the Incorrect_Cap.
5. IF the Correctness_Checker raises an error or fails to load while a Submission is scored in the Arena, THEN THE App SHALL still display the Arena result with Correctness_Status `unverified`, the "Unverified" badge, IVS computed from that `unverified` status, and a warning indicating that correctness verification failed, and SHALL allow the Submission to be stored with Correctness_Status `unverified`.
6. IF IVS computation with `diverge/value.py` raises an error for a Submission, THEN THE App SHALL display Divergence_Score_D and the Legacy_Composite with an error message indicating that IVS could not be computed, SHALL not display any IVS value for that Submission, and THE Store SHALL not write a record containing a partial or placeholder IVS for that Submission.
7. THE App SHALL label IVS and each IVS dimension (Semantic_Novelty, Reasoning_Quality, Evidence_Quality, Marginal_Value) as "estimate" wherever they appear in the Arena, Leaderboard and Training Data views, and SHALL label Marginal_Value "estimated proxy, not measured learning gain".
8. THE App SHALL run within the four-page navigation of Requirement 15, run in the existing single Streamlit server, and score Submissions, including the Correctness_Checker step, with local models and local files only, making zero network calls.

### Requirement 13: Training-grade eligibility

**User Story:** As an ML engineer, I want training eligibility to depend on value and correctness, so that a highly divergent but wrong answer never enters training data by default.

#### Acceptance Criteria

1. THE Store SHALL classify a record as Training_Grade only when the record's IVS ≥ IVS_Threshold and its Correctness_Status is not `verified_incorrect`.
2. THE Store SHALL classify a record as Training_Grade without Divergence_Score_D or the Legacy_Composite as inputs.
3. WHEN the Leaderboard is displayed, THE App SHALL compute each user's training-grade count as the number of that user's Training_Grade records.
4. WHEN the Training Data view is opened, THE App SHALL default its minimum IVS filter to IVS_Threshold and expose IVS_Threshold as an adjustable control in the range 0–100.
5. WHEN a legacy record without IVS or Correctness_Status is loaded, THE Store SHALL either compute IVS from the record's stored `components` with Correctness_Status `unverified` in memory, or treat IVS as empty and the record as not Training_Grade, leaving `data/submissions.jsonl` unchanged in both cases.
6. WHEN legacy records are loaded, THE App SHALL load, render and export all 11 records present in `data/submissions.jsonl` before this feature.

### Requirement 14: Objectively verifiable questions with exact expected answers

**User Story:** As a researcher, I want some questions with an exact, citable answer, so that correctness can be checked by a human reference instead of the AI's opinion.

#### Acceptance Criteria

1. THE feature SHALL add at least 4 questions with ids `q11` onward to `diverge/questions.py`, each with Verifiable_Flag true and an objectively checkable short answer, covering the bat-and-ball problem, the Monty Hall problem, a base-rate false-positive problem and a birthday-paradox count, while leaving `q01` to `q10` unchanged.
2. THE feature SHALL ship a human-authored Reference_Key with an Expected_Answer, at least 1 Accepted_Claim and a source citation for every question with Verifiable_Flag true, each passing every check in Requirement 2 criterion 2 with 0 load warnings.
3. WHEN a Submission is scored for a question whose Reference_Key has an Expected_Answer and an Extracted_Answer exists, THE Correctness_Checker SHALL set Correctness_Status to `verified_correct` if the Extracted_Answer matches the expected value (within tolerance for `numeric`; containing an accepted alias as a whole word and no known wrong alias for `string`) and to `verified_incorrect` otherwise.
4. IF a Submission is for a question whose Reference_Key has an Expected_Answer and no Extracted_Answer exists, THEN THE Correctness_Checker SHALL set Correctness_Status to `unverified`.
5. WHEN a Reference_Key has an Expected_Answer, THE Correctness_Checker SHALL decide Correctness_Status by criteria 3 and 4 only, SHALL record the Accepted_Claim and Rejected_Claim NLI results of Requirement 3 as supplementary verification evidence, and SHALL record as deciding verification evidence the expected value, the Extracted_Answer and the Expected_Answer citation.
6. WHEN Correctness_Status is decided by an Expected_Answer, THE Scorer SHALL state in the Correctness_Status sentence the expected value, the Extracted_Answer and the citation, and SHALL skip the "borderline verification" check of Requirement 7 criterion 5.
7. THE Correctness_Checker SHALL decide Correctness_Status for questions with an Expected_Answer without Baselines or any LLM output as inputs.
8. THE feature SHALL ship cached Baselines in `data/baselines.json` for every question added under criterion 1, generated offline with the existing generator in `diverge/baselines.py`, so that the App generates no Baselines during page render.
9. IF a question added under criterion 1 has fewer than 2 cached Baselines, THEN THE App SHALL still score Submissions for that question and THE Scorer SHALL apply Requirement 7 criterion 7.
10. WHEN a question with Verifiable_Flag true is shown in the Arena, THE App SHALL display a hint asking the contributor to state the final answer first in the answer field.

### Requirement 15: Navigation and Home page

**User Story:** As a first-time visitor, I want a clear landing page and simple navigation, so that I understand the mission and can start a challenge in one click.

#### Acceptance Criteria

1. THE App SHALL provide exactly four primary navigation entries, in this order: HOME, CHALLENGES, MY IMPACT, ABOUT.
2. THE App SHALL keep the Consensus Breaker and the Export_Panel reachable from a non-primary location (a secondary link, expander or sub-section of a primary page).
3. THE Home_Page SHALL show the hero headline "Make AI smarter. Not bigger.", the subtitle "AI's next breakthrough might not require more computing power. It might require better information." and the supporting statement "Discover what AI doesn't know. Contribute better reasoning. Help us investigate whether higher-value training data can reduce the energy required to develop better AI."
4. WHEN the user activates the "Start a Challenge" call to action, THE App SHALL open the Challenges_Page.
5. WHEN the user activates the "How It Works" call to action, THE App SHALL scroll to or expand the steps section.
6. THE Home_Page SHALL show four steps labeled "01 Find AI's blind spots", "02 Contribute intelligence", "03 Measure its value" and "04 Test AI efficiency".
7. THE Home_Page SHALL show the climate link (better data → less compute → less energy) labeled "hypothesis".
8. WHERE Experiment_Results exist, THE Home_Page SHALL show an experiment summary of at most 4 values, each read from Experiment_Results and accompanied by its provenance (file, run date, seed count).
9. IF Experiment_Results are missing or unreadable, THEN THE Home_Page SHALL show the experiment summary as "pending" without any numeric value.
10. THE Home_Page SHALL show no data tables and no energy or CO₂ value presented as measured.

### Requirement 16: Challenges page and Challenge workspace

**User Story:** As a contributor, I want to browse challenges like coding problems and answer one in a focused workspace, so that I can contribute reasoning and see a transparent evaluation.

#### Acceptance Criteria

1. THE Challenges_Page SHALL list every question in `diverge/questions.py` (`q01` to `q10` and the Requirement 14 questions) with its title, difficulty, category, Verification_Type and Solved status for the current Display_Name.
2. WHILE no Display_Name is set, THE Challenges_Page SHALL show every question as not Solved.
3. WHEN the user selects a challenge, THE App SHALL open the Challenge_Workspace in two columns.
4. THE Challenge_Workspace left column SHALL show the challenge title and statement, a summary of 1 cached Baseline labeled "AI baseline approach", the question's Consensus_Assumptions labeled "assumptions to investigate (hypotheses, not errors)", and each cached Baseline in an expandable section.
5. THE Challenge_Workspace right column SHALL show inputs for Display_Name, answer, reasoning and "evidence or proposed verification test", a training-use consent checkbox unchecked by default, and a "Submit for Evaluation" button.
6. IF the Display_Name is outside 1–40 characters after trimming, contains "@", or the answer is empty after trimming, THEN THE App SHALL show a message naming the invalid field, keep the entered text, and neither score nor store the Submission.
7. WHEN a valid Submission is submitted, THE App SHALL store the evidence text with the reasoning text, store Training_Consent from the checkbox, and score the Submission per Requirement 12.
8. WHEN a Submission has been scored, THE Challenge_Workspace SHALL show Divergence Score (D), Reasoning Quality (Q), Estimated Learning Value (IVS), the Correctness_Status badge and an explanation of at most 4 sentences.
9. THE App SHALL take every number shown in the Challenge_Workspace from the outputs of `scoring.score_submission`, `value.intelligence_value` and the Correctness_Checker, with no recalculation in UI code.
10. THE Challenge_Workspace SHALL provide an expandable "How scoring works" section that explains D, Q and IVS; the Quality_Gate; that correctness comes only from human-authored Reference_Keys; that open-ended questions are `unverified` with uncertainty shown; and that Estimated_Learning_Value is not a measured learning gain.
11. THE "How scoring works" section SHALL state that only automated local evaluators (heuristic rubric, NLI, perplexity) are used and that no independent human or multi-model review panel exists yet.
12. WHEN a question with Verifiable_Flag true is open in the Challenge_Workspace, THE App SHALL show the Requirement 14 criterion 10 hint.

### Requirement 17: Three-metric integrity

**User Story:** As a researcher, I want D, Q and IVS kept separate with explicit rules, so that being different is never mistaken for being valuable.

#### Acceptance Criteria

1. THE Scorer SHALL compute Reasoning_Quality and IVS without Divergence_Score_D as an input.
2. THE Store SHALL classify Training_Grade from IVS, IVS_Threshold, Correctness_Status, Training_Consent and the demo-seed flag only, without Divergence_Score_D as an input.
3. FOR ALL records with Correctness_Status `verified_incorrect` and any Divergence_Score_D in 0–100, THE Store SHALL classify the record as not Training_Grade.
4. WHEN a `verified_correct` Submission with Reasoning_Quality ≥ Floor_Quality_Threshold is scored, THE Scorer SHALL produce IVS ≥ Conventional_Floor, independent of Divergence_Score_D.
5. WHEN a Submission for an Open-ended question is displayed, THE Challenge_Workspace SHALL show the badge "Unverified" and the Uncertainty_Level.
6. THE App SHALL describe agreement or disagreement with the AI Baselines only as similarity or difference, using no UI copy that calls such agreement or disagreement correct or incorrect.
7. WHEN the demo seeds of `diverge/seed.py` are scored, THE Store SHALL classify the `demo:noise` and `demo:offtopic` records as not Training_Grade.

### Requirement 18: My Impact page

**User Story:** As a contributor, I want to see my contributions and how they feed the efficiency research, so that I understand my impact without inflated claims.

#### Acceptance Criteria

1. WHEN a Display_Name is set in the session or entered on the My_Impact_Page, THE My_Impact_Page SHALL show, for that Display_Name: challenges completed (distinct Solved questions), contributions submitted, verified corrections (count of `verified_correct` Submissions to questions with Verifiable_Flag true), contributions accepted for training (Training_Grade records), average Divergence_Score_D labeled "how different, not how good", and average IVS labeled "Estimated Learning Value (estimate)".
2. IF the Display_Name has no stored Submissions, THEN THE My_Impact_Page SHALL show 0 for each count and "no data" for each average.
3. THE My_Impact_Page SHALL show a section titled "Your contribution to greener AI" with the pathway Your reasoning → Curated training example → Model evaluation → Measured compute efficiency → Potential electricity savings, each step showing the user's count from the Store or the text "not yet measured".
4. THE My_Impact_Page SHALL state that energy savings cannot currently be attributed to individual contributors and SHALL show no per-user kWh or CO₂ value.
5. THE My_Impact_Page SHALL show a leaderboard ranked by verified-correct count, then Training_Grade count, then mean Reasoning_Quality, all descending, with Divergence_Score_D used as no ranking key and demo-seed entries labeled "demo".
6. WHEN the My_Impact_Page renders, THE App SHALL read the Store at render time, so that a Submission stored earlier in the session appears in the counts and leaderboard.

### Requirement 19: About page, experiment and energy scenarios

**User Story:** As a judge or researcher, I want the mission, method and evidence presented honestly, so that I can tell measured results from hypotheses.

#### Acceptance Criteria

1. THE About_Page SHALL show these texts exactly: Vision "A future where AI advances through better intelligence, not just greater computation."; Mission "To discover, evaluate, and unlock valuable human reasoning that helps AI learn more efficiently."; Goal "To test whether high-information-value training data can improve AI capability per unit of compute—and translate those improvements into measurable energy efficiency."; Belief "The most valuable idea isn't always the most popular one. It's the one that teaches us something new."
2. THE About_Page SHALL show a methodology section covering: why AI training uses electricity; why data quality may affect learning efficiency; why unconventional reasoning is not automatically useful; how DIVERGE reduces evaluation bias; and how actual learning gains will be measured.
3. THE About_Page SHALL show the experiment in three visually separated blocks labeled "Measured performance", "Estimated energy impact" and "Hypothetical scenarios".
4. WHERE Experiment_Results exist, THE "Measured performance" block SHALL show values read from Experiment_Results only, with config, seeds, dataset, model and run date, and SHALL state that divergent selection did not beat random selection and that no difference reached p < 0.05.
5. IF Experiment_Results are missing, THEN THE "Measured performance" block SHALL show "pending" and the command to run `diverge/experiment.py`.
6. THE Energy_Calculator SHALL compute baseline energy (kWh) = Average Power (kW) × Duration (hours), scenario duration = baseline duration × (1 − assumed compute reduction), and scenario savings (kWh) = baseline energy − scenario energy, with every input labeled "assumption".
7. THE Energy_Calculator SHALL default the assumed compute reduction to 0%, so that scenario savings are 0 kWh until the user changes it.
8. THE "Hypothetical scenarios" block SHALL state that reduced training compute does not automatically reduce total data-center electricity consumption, naming rebound effects, utilization and inference demand.
9. THE About_Page SHALL show no energy, kWh, CO₂ or cost value labeled as measured unless it is read from Experiment_Results.
10. THE Export_Panel SHALL keep the 3 export formats and the Requirement 8 criterion 3 and Requirement 13 filters, default the minimum IVS filter to IVS_Threshold, and SHALL export only the records the user selected.

### Requirement 20: Visual design

**User Story:** As a visitor, I want a calm, modern interface, so that the content is easy to read and the product feels focused.

#### Acceptance Criteria

1. THE App SHALL use a deep charcoal and purple color scheme with electric lime accents and minimal typography.
2. THE Challenges_Page SHALL present challenges as cards or rows showing only the Requirement 16 criterion 1 fields.
3. THE App SHALL lay out multi-column views with Streamlit columns, so that columns stack on narrow screens.
4. THE App SHALL convey every status (Correctness_Status, Solved, measured vs hypothesis) with a text label, with color as a secondary cue only.
