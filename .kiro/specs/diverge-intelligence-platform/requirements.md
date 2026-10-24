# Requirements Document

## Introduction

Upgrade of the existing DIVERGE Streamlit app (`app.py`, `diverge/`) into the DIVERGE Intelligence Platform: "Find what AI doesn't know." Hypothesis under test: the most valuable training data has the greatest marginal learning value, not the most conventional or the most unusual content. Climate link (hypothesis, not claim): better data efficiency could reduce the compute and electricity needed for a target capability.

Scope is limited to what can ship today by reusing existing modules: `models.py` (local MiniLM, DeBERTa NLI, Qwen2.5-0.5B), `baselines.py`, `scoring.py`, `store.py`, `questions.py`, `seed.py`, `experiment.py`. `engine/` and `unbound/` stay untouched. No API keys, no rebuild.

## Glossary

- **App**: The DIVERGE Streamlit application in `app.py`.
- **Baseline**: One of the cached AI answers per question in `data/baselines.json`.
- **Consensus_Assumption**: A sentence recurring across at least 2 Baselines (output of `consensus_assumptions()`).
- **Blind_Spot_Analyzer**: New module that computes Baseline disagreement and suspected weaknesses per question.
- **Disagreement_Index**: 0-100 value derived from pairwise Baseline embedding dissimilarity and NLI contradiction.
- **Causal_Marker**: A case-insensitive whole-word match of "because", "therefore", "thus", "so", "since", "due to", "as a result", "leads to", or "causes" in a Baseline's text.
- **Scorer**: The scoring module producing the Divergence_Score and Intelligence_Value.
- **Divergence_Score (D)**: 0.50·S + 0.30·N + 0.20·A, each component 0-100.
- **Intelligence_Value (V)**: 0.35·Q + 0.25·E + 0.40·G, each component 0-100.
- **Marginal_Gain_Proxy (G)**: Estimated learning gain from transparent proxies (information gain vs existing pool, claim novelty vs dataset, quality gating); an estimate, not a measurement.
- **Ungated_G**: The information gain value before the quality gate is applied, 0-100.
- **Gated_G**: Ungated_G after applying the quality gate factor; the G used in V.
- **Blind_Evaluator**: Evaluation path that scores quality on text stripped of human/AI origin.
- **Evaluator_Disagreement**: Absolute difference (0-100) between the heuristic rubric evaluator score and the local-model evaluator score for Q.
- **Verification_Status**: One of `Verified useful`, `Promising but unverified`, `Incorrect or unsupported`, `Needs human review`.
- **Store**: The JSONL submission store in `diverge/store.py`.
- **Experiment_Runner**: `diverge/experiment.py`.
- **Compute_Efficiency**: (held-out accuracy − 0.20 chance) / (training FLOPs / 10^15).
- **Energy_Calculator**: Climate tab widget computing energy = compute time × hardware power × (1 − efficiency improvement).

## Requirements

### Requirement 1: AI Blind Spots Dashboard

**User Story:** As a contributor, I want to see where AI baselines disagree or look weak, so that I know which questions to challenge.

#### Acceptance Criteria

1. THE App SHALL display a Blind Spots tab as the first tab. For each question, the tab SHALL list its 3 cached Baselines, up to 5 Consensus_Assumptions with their support count (2 or 3 means the claim appears in that many Baselines), its Disagreement_Index as an integer from 0 to 100, and 0 or more suspected weaknesses.
2. THE Blind_Spot_Analyzer SHALL compute the Disagreement_Index as round(100 × (0.5 × D_cos + 0.5 × C_max)). D_cos is the mean of (1 − cosine similarity) over the 3 Baseline pairs using MiniLM embeddings, clipped to 0–1. C_max is the maximum DeBERTa NLI contradiction probability over all cross-Baseline sentence pairs, evaluated in both directions.
3. THE Blind_Spot_Analyzer SHALL flag a Baseline as a reasoning gap if either condition holds: (a) it contains 0 Causal_Markers, where a Causal_Marker is a case-insensitive whole-word match of "because", "therefore", "thus", "so", "since", "due to", "as a result", "leads to", or "causes"; or (b) at least one of its sentences has an NLI contradiction probability strictly greater than 0.5 against a sentence of another Baseline. Each flag SHALL name which condition triggered it.
4. THE App SHALL show the text "Hypothesis: unverified" next to every suspected weakness. A question with 0 suspected weaknesses SHALL show a "No suspected weaknesses" indicator instead.
5. WHEN the user clicks a question's "Challenge" button, THE App SHALL switch to the Submission view with that question preselected. Any other Submission input the user has already entered SHALL stay unchanged.
6. THE App SHALL sort questions by Disagreement_Index in descending order by default. Questions with equal Disagreement_Index SHALL be ordered alphabetically by question text.
7. IF a question has no cached Baselines, or fewer than 3, THEN THE App SHALL list that question with a "Baselines unavailable" indicator. It SHALL not show a Disagreement_Index or suspected weaknesses for that question, SHALL place it after all scored questions, and SHALL not generate Baselines while the tab renders.
8. IF the MiniLM embedding model or the DeBERTa NLI model fails to load, THEN THE App SHALL show an error message in the Blind Spots tab naming the unavailable model. It SHALL keep showing Baselines and Consensus_Assumptions where they can still be computed, and SHALL show the Disagreement_Index and NLI-based weaknesses as "unavailable".

### Requirement 2: Human Reasoning Submission

**User Story:** As a contributor, I want to challenge a specific AI assumption with my own reasoning and evidence, so that my contribution is evaluated on its merits.

#### Acceptance Criteria

1. WHEN the user selects a question, THE App SHALL display all Baselines for that question and a list of its Consensus_Assumptions from which the user can select zero or more to challenge.
2. THE App SHALL provide three separate text inputs labeled answer, reasoning, and evidence/counterexample, each accepting up to 3,000 characters, where reasoning and evidence/counterexample are optional.
3. WHEN the user submits a submission that passes validation, THE Scorer SHALL compute D and V for the submission, and THE Store SHALL append one record to the JSONL store containing the user name, question, answer, reasoning, evidence/counterexample, the selected challenged Consensus_Assumptions (an empty list if none were selected), D, V, and a submission timestamp.
4. IF the answer field contains fewer than 15 whitespace-separated words after trimming leading and trailing whitespace, THEN THE App SHALL reject the submission without invoking THE Scorer or writing to THE Store, retain all entered input values, and display a message stating that the minimum answer length is 15 words.
5. IF the user name is empty or no question is selected at submit time, THEN THE App SHALL reject the submission without invoking THE Scorer or writing to THE Store, retain all entered input values, and display a message identifying the missing field.
6. IF THE Scorer raises an error or THE Store fails to write the record, THEN THE App SHALL display an error message indicating that the submission was not saved, retain all entered input values, and leave the JSONL store without a partial record for that submission.
7. WHEN a submission is scored, THE App SHALL display the submitted answer alongside the Baseline with the highest cosine similarity to it, plus up to 3 prior submissions for the same question with the highest cosine similarity, each shown with its similarity value rounded to 2 decimal places; IF fewer than 3 prior submissions exist for that question, THEN THE App SHALL show all that exist, and IF none exist, THEN THE App SHALL show a message indicating no prior submissions.
8. WHEN the leaderboard is displayed, THE App SHALL show one row per user, ranked in descending order of the mean of (D + V) / 2 across that user's submissions, with mean D and mean V displayed as separate columns and ties ordered alphabetically by user name.

### Requirement 3: Divergence Score

**User Story:** As a contributor, I want a transparent originality score, so that I understand how far my reasoning departs from AI consensus.

#### Acceptance Criteria

1. WHEN a submission is scored, THE Scorer SHALL compute S (semantic divergence: 1 minus the calibrated maximum cosine similarity between the submission and any Baseline), N (conceptual novelty: mean calibrated claim novelty of the submission's sentences against all sentences in the Baselines and prior submissions for the same question), and A (assumption challenge: support-weighted mean of the maximum NLI contradiction probability per Consensus_Assumption), each scaled from 0-1 to 0-100 and rounded to 1 decimal place.
2. WHEN S, N, and A have been computed, THE Scorer SHALL compute D = 0.50·S + 0.30·N + 0.20·A, so that D falls in the range 0-100, rounded to 1 decimal place.
3. WHERE the user selected one or more challenged assumptions, THE Scorer SHALL compute A over the selected Consensus_Assumptions only, using each selected assumption's support as its weight.
4. IF the user selected zero challenged assumptions, THEN THE Scorer SHALL compute A over all Consensus_Assumptions for the question.
5. IF no Consensus_Assumptions exist for the question or the submission has zero extractable claims, THEN THE Scorer SHALL set A to 0 and N to 0 respectively, still compute D, and mark the affected component as "not applicable" in the score breakdown.
6. THE Scorer SHALL compute D independently of Q, E, and G, so that changing Q, E, or G with the submission text unchanged leaves D, S, N, and A unchanged.
7. WHEN the submission is scored with identical text, Baselines, and prior submissions, THE Scorer SHALL return the same S, N, A, and D values (within ±0.1).
8. WHEN D is displayed, THE App SHALL show for each of S, N, and A the component value (0-100), its weight (0.50, 0.30, 0.20), its contribution to D (weight × value), and a single-sentence explanation that cites the measured quantity behind the value (closest Baseline similarity for S, count of novel claims out of total claims for N, the most contradicted Consensus_Assumption for A), with the three contributions summing to the displayed D within ±0.1.

### Requirement 4: Intelligence Value Score

**User Story:** As a dataset curator, I want a usefulness score separate from originality, so that disagreement alone is never rewarded.

#### Acceptance Criteria

1. WHEN a submission is scored, THE Scorer SHALL compute Q as the existing reasoning quality value (relevance, fluency, self-consistency, structure, depth) on its 0-1 scale multiplied by 100, giving a value from 0 to 100.
2. WHEN a submission is scored for a question that has no objective answer, THE Scorer SHALL compute E as the existing evidence heuristics value on its 0-1 scale multiplied by 100, giving a value from 0 to 100.
3. WHEN a submission is scored for a question that has an objective answer, THE Scorer SHALL set E to 100 if the submission's answer matches the objective answer and to 0 if it does not.
4. WHEN a submission is scored, THE Scorer SHALL compute the Ungated_G as the existing information gain value (embedding residual against the pool of baselines and prior submissions) on its 0-1 scale multiplied by 100, giving a value from 0 to 100.
5. WHEN Q, E and G have been computed, THE Scorer SHALL compute V = 0.35·Q + 0.25·E + 0.40·G using the Gated_G, round V to 1 decimal place, and keep V between 0 and 100 inclusive.
6. IF Q is below 25, THEN THE Scorer SHALL set the Gated_G to 0. IF Q is 25 or higher, THEN THE Scorer SHALL set the Gated_G to the Ungated_G multiplied by the quality gate factor min(1, (Q − 25) / 35), so the Gated_G never exceeds the Ungated_G.
7. IF Q, E or G cannot be computed because a scoring model fails or is unavailable, THEN THE Scorer SHALL leave V unset for that submission. THE App SHALL then show a message that V is unavailable and still show any D value that was computed.
8. WHEN G is displayed anywhere in the App, THE App SHALL show the label "estimated proxy, not measured" next to that G value.
9. WHEN a scored submission is displayed, THE App SHALL show D and V as two separate visualizations and show the label "experimental assumptions" next to the V weights (0.35, 0.25, 0.40).
10. FOR ALL submissions in a test set of at least 10 off-topic or random-text inputs, THE Scorer SHALL produce V below 30 for every input, even when D is above 50 (the high-D/low-V property). Off-topic or random text means either a random sequence of dictionary words or a fluent passage about an unrelated topic, where each input's question relevance score is below 0.2 on the 0-1 scale.

### Requirement 5: Bias-Resistant Evaluation

**User Story:** As a researcher, I want evaluation that does not favor or penalize text for agreeing with AI, so that valid contrarian reasoning survives.

#### Acceptance Criteria

1. THE Blind_Evaluator SHALL compute Q only from the question text and the submission's answer and reasoning text, with author name, origin, submission timestamp, and any other submission metadata excluded, so that two submissions with identical text and different metadata receive identical Q values.
2. THE Scorer SHALL compute novelty components (semantic divergence, claim novelty, assumption challenge, information gain) without using reference-answer correctness, E, or any measure of agreement with Consensus_Assumptions as an input.
3. THE Scorer SHALL compute Q on a 0–100 scale from at least 2 independent evaluators (a heuristic rubric evaluator and a local-model evaluator using Qwen2.5-0.5B perplexity and DeBERTa NLI signals), each producing a 0–100 score, and SHALL record the absolute difference between the 2 evaluator scores (0–100) as Evaluator_Disagreement.
4. WHERE a question has an objective reference answer, THE Scorer SHALL compute E from correctness against the reference answer and SHALL mark the submission as verified correct or verified incorrect; WHERE a question has no objective reference answer, THE Scorer SHALL mark correctness as unverified.
5. THE Scorer SHALL assign exactly one Verification_Status per submission, evaluating rules in this order and applying the first match: (a) `Incorrect or unsupported` when verified incorrect; (b) `Verified useful` when verified correct and V ≥ 60; (c) `Needs human review` when Evaluator_Disagreement > 25 or (D ≥ 60 and Q < 50); (d) `Incorrect or unsupported` when Q < 25; (e) otherwise `Promising but unverified`.
6. IF a submission has Q ≥ 25, has no objective reference answer, and has a DeBERTa NLI contradiction probability ≥ 0.5 against at least 1 Consensus_Assumption, THEN THE Scorer SHALL assign a Verification_Status other than `Incorrect or unsupported`.
7. WHEN the App displays V for a submission, THE App SHALL display the Evaluator_Disagreement value (0–100, integer) adjacent to V in the same view.
8. IF one of the evaluators fails to produce a score (local model load or inference failure), THEN THE Scorer SHALL compute Q from the remaining evaluator, SHALL assign `Needs human review` unless rule 5(a) applies, and THE App SHALL display an indication that Evaluator_Disagreement is unavailable in place of the disagreement value.

### Requirement 6: Training Data Engine

**User Story:** As an ML engineer, I want consented, filtered, well-documented training records, so that I can use them responsibly.

#### Acceptance Criteria

1. WHEN a submission is accepted, THE Store SHALL append one JSONL record containing the question, Baselines, answer, reasoning, challenged assumptions, evidence, D, V, components, Verification_Status, consent flag, display name, submission timestamp, unique record identifier, and evaluation metadata.
2. THE App SHALL display a consent checkbox for training use that is unchecked by default on every new submission form.
3. IF the user submits while the consent checkbox is unchecked, THEN THE App SHALL reject the submission, write no record to the Store, display an error message indicating consent is required, and retain the entered answer and reasoning in the form.
4. THE Store SHALL store only a user-chosen display name of 1 to 40 characters as the user identifier, and SHALL store no email address, IP address, or other personal identifier.
5. IF the entered display name is empty, exceeds 40 characters, or contains an "@" character, THEN THE App SHALL reject the submission, write no record to the Store, and display an error message indicating the display name is invalid.
6. WHEN the user sets export filters, THE App SHALL include only records whose V is greater than or equal to the minimum V, whose D is greater than or equal to the minimum D, and whose Verification_Status is in the selected set, where minimum V and minimum D each accept values from 0 to 100 inclusive.
7. WHEN the user exports in full, chat, or DPO format, THE Store SHALL produce a JSONL file with exactly one record per line containing only records that pass the active filters and have consent set to true.
8. IF a record selected for DPO export has no Baseline, THEN THE Store SHALL omit that record from the DPO export.
9. IF no records pass the active filters and consent check, THEN THE App SHALL disable the download action and display a message indicating zero records match.
10. FOR ALL exported records in every format, parsing each JSONL line SHALL yield an object whose D, V, and Verification_Status equal the stored values for that record identifier (round-trip property).
11. WHEN a legacy record lacking D, V, or consent fields is loaded, THE Store SHALL rescore missing D and V from the stored answer and Baselines, set consent to true for demo-seed records, and set consent to false for all other records.
12. IF a line in the Store cannot be parsed as JSON, THEN THE Store SHALL skip that line, continue loading the remaining records, and exclude the skipped line from all exports.

### Requirement 7: Learning Efficiency Experiment

**User Story:** As a judge, I want an honest experiment comparing data selection strategies, so that the hypothesis is tested rather than asserted.

#### Acceptance Criteria

1. THE Experiment_Runner SHALL support arms conventional (A), random (B), and DIVERGE-curated (C), where C selects each next training question by a marginal-value score combining coverage (distance from already-selected questions in the frozen MiniLM embedding space) and typicality (similarity to the training-pool distribution), and does not rank by maximal divergence alone.
2. THE Experiment_Runner SHALL use the same frozen all-MiniLM-L6-v2 embeddings, the same 2-layer MLP architecture, the same hyperparameters (hidden size, epochs, batch size, learning rate, weight decay), the same budgets (100, 200, 400, 800, 1600, 3200 questions) and the same seeds (0 to 4) for every arm.
3. THE Experiment_Runner SHALL evaluate every run on the CommonsenseQA validation split (1,221 questions), and SHALL exclude every validation question from training and from selection input for every arm.
4. WHEN a run (arm, budget, seed) completes, THE Experiment_Runner SHALL record held-out accuracy, training steps, analytic training FLOPs, wall-clock training seconds, and Compute_Efficiency = (accuracy − 0.20 chance) / (training FLOPs / 10^15).
5. WHEN all runs complete, THE Experiment_Runner SHALL report budget-to-target-accuracy for targets 0.30, 0.33, and 0.36 as the smallest budget whose 5-seed mean accuracy is ≥ the target, recording "not reached" when no budget reaches it.
6. WHEN all runs complete, THE Experiment_Runner SHALL compute, per budget, the mean accuracy difference of C versus A and C versus B together with a two-sided paired sign-flip permutation p-value over the 5 seeds using 10,000 permutations.
7. WHEN the App loads `data/experiment/results.json`, THE App SHALL display the measured per-arm, per-budget mean accuracy, Compute_Efficiency, budget-to-target-accuracy, and p-values from that file, and SHALL display a statement that no difference is statistically significant when every displayed p-value is ≥ 0.05.
8. IF results for an arm have not been executed, THEN THE App SHALL label that arm "Not yet run" and display no numbers for the arm.
9. IF `data/experiment/results.json` is missing or cannot be parsed, THEN THE App SHALL label every arm "Not yet run", display a message indicating results are unavailable, and continue rendering the rest of the App without error.
10. THE App SHALL retain the existing measured divergent-arm results, labeled as a prior run and displayed separately from arm C results.

### Requirement 8: Climate Impact Dashboard

**User Story:** As a climate-track judge, I want the compute-energy link presented with evidence separated from projection, so that I can assess the claim's credibility.

#### Acceptance Criteria

1. WHEN the climate dashboard is opened, THE App SHALL display the causal chain as 5 ordered nodes (human reasoning → higher-information data → learning efficiency → potentially reduced compute → potentially lower data-center electricity), with each link labeled either "measured" or "hypothesis", and the last 2 links always labeled "hypothesis".
2. WHEN the climate dashboard is opened, THE App SHALL display experiment results only in a section labeled "Measured" and calculator outputs only in a separate section labeled "Hypothetical projection", with no calculator output appearing in the "Measured" section.
3. WHEN the user changes any Energy_Calculator input, THE Energy_Calculator SHALL accept baseline compute (1 to 1,000,000,000 GPU-hours), efficiency improvement (0 to 100 %), hardware power (0.01 to 100 kW per GPU), and grid carbon intensity (0 to 2 kg CO₂/kWh), and SHALL display energy saved in kWh as GPU-hours × (efficiency % ÷ 100) × power kW, and emissions avoided in kg CO₂ as energy saved × carbon intensity.
4. WHEN experiment results are available and the measured efficiency improvement is greater than 0 %, THE Energy_Calculator SHALL default the efficiency improvement input to that measured value labeled "measured", and IF the user changes it, THEN THE Energy_Calculator SHALL label the value "assumed".
5. THE App SHALL display, within the "Hypothetical projection" section and visible without user interaction, a statement that no real-world energy savings have been demonstrated.
6. IF an Energy_Calculator input is empty, non-numeric, or outside its stated range, THEN THE Energy_Calculator SHALL show an error message naming the invalid input, display no energy-saved or emissions-avoided value, and keep the other inputs unchanged.
7. IF no experiment results are available, or the measured efficiency improvement is 0 % or less, THEN THE Energy_Calculator SHALL default the efficiency improvement to 0 %, label it "assumed", and display a notice that no positive measured improvement exists.

### Requirement 9: Visual Design

**User Story:** As a demo viewer, I want a distinctive, polished interface, so that the product feels credible.

#### Acceptance Criteria

1. THE App SHALL render every tab using the existing dark theme, with deep purple #8B5CF6 and electric lime #C6FF3D as the only accent colors for highlights, active tab indicators, and chart series.
2. THE App SHALL render all headings, tab labels, and headline metric values in a custom display font that differs from the body text font, and SHALL fall back to a sans-serif font IF the custom display font fails to load.
3. WHEN an evaluation result is displayed, THE App SHALL render the D and V component scores as a radar or bar chart with each component labeled by name and its numeric value on a 0 to 100 scale.
4. WHEN an evaluation result is displayed, THE App SHALL render the AI baseline answers and the human submission in side-by-side columns within the same view, each labeled with its source (AI or human).
5. THE App SHALL provide exactly 7 tabs in this left-to-right order: Blind Spots, Challenge, Leaderboard, Training Data, Experiment, Climate Impact, Method, with Blind Spots selected on initial load.
6. WHEN the App starts and the Store contains 0 submissions, THE App SHALL load the seed submissions into the Store with each one flagged as demo, and SHALL display a "demo" label next to each seeded entry wherever it appears in the Leaderboard and Training Data tabs.
7. WHEN the App starts and the Store contains 1 or more submissions, THE App SHALL leave the Store unchanged and SHALL NOT load seed submissions.
8. IF seed submissions fail to load, THEN THE App SHALL keep the Store empty, show an empty-state message in the Leaderboard and Training Data tabs, and keep all 7 tabs usable.
