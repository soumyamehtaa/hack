# Design Document: Bias-Resistant Divergence Scoring

## Overview

This design extends the existing DIVERGE scorer instead of replacing it. One new orchestration function, `diverge/evaluate.py::evaluate()`, calls the unchanged `scoring.score_submission`, a new human-reference `correctness` checker, the existing `value.intelligence_value`, and a new uncertainty step. It returns one additive `assessment` dict that gets stored next to the existing `evaluation` field. `app.py` is rebuilt from seven tabs into four pages (HOME, CHALLENGES, MY IMPACT, ABOUT), and it reuses the existing helpers, CSS palette and experiment functions.

Design principles, all taken from the workspace rules:

- Legacy fields stay identical. `score_submission` keeps its signature and every existing output field. It gains one new key, `divergence_score_d`, and a VERSION bump to `diverge-score-0.2`.
- Correctness comes only from `data/references.json`, which is human-authored. Baselines are never passed to the correctness checker.
- Every displayed number comes from stored data or from a reproducible run. Values are labeled `estimate`, `measured` or `hypothesis`.
- `data/submissions.jsonl` (11 records) is never rewritten. Legacy records are enriched in memory only.
- Everything runs locally with no network calls, targets Python 3.9 (`from __future__ import annotations`), and uses the one existing Streamlit server on :8501, which hot-reloads. `engine/` and `unbound/` stay untouched.

Scope for the hackathon day. Rows marked deferred are not implemented today:

| Area | Today | Deferred |
|---|---|---|
| Req 1, 4, 11, 13, 17 | full | – |
| Req 2, 3, 14 correctness | Expected_Answer exact match decides; NLI claims decide open-ended keys | – |
| Req 7 uncertainty | leave-one-out over baselines + correctness swap via `value.combine` | – |
| Req 9 reproducibility | Evaluation_Config dict, Config_Hash, prior-pool hashes, environment record, seeding, validation, determinism test | 9.4/9.5/9.9/9.10 rescore-and-compare flow (`rescore()` with "config changed" / "environment changed" / "not reproducible" flags); 9.1 moving `scoring.py` constants physically into the config (config mirrors them by import instead) |
| Req 15–20 UI | full four-page app | – |

## Architecture

```mermaid
flowchart LR
  subgraph data
    SUB[(submissions.jsonl)]
    BL[(baselines.json)]
    REF[(references.json NEW)]
    EXP[(experiment/results.json)]
  end
  Q[questions.py +q11-q14] --> APP
  CFG[config.py NEW<br/>EVAL_CONFIG + hash] --> EV
  APP[app.py<br/>router: HOME / CHALLENGES / MY IMPACT / ABOUT] --> EV[evaluate.py NEW<br/>evaluate()]
  EV --> SC[scoring.py<br/>score_submission +D]
  EV --> CC[correctness.py NEW<br/>Checker]
  EV --> VA[value.py<br/>combine / intelligence_value]
  EV --> UN[uncertainty in evaluate.py]
  SC --> M[models.py]
  CC --> M
  CC --> REF
  APP --> ST[store.py<br/>+view, training_grade, export fields]
  ST --> SUB
  APP --> BL
  APP --> XP[experiment.py<br/>summarize / significance / compute_to_target] --> EXP
  GEN[diverge/gen_baselines.py NEW, offline] --> BL
```

New modules: `diverge/config.py`, `diverge/correctness.py`, `diverge/evaluate.py`, `diverge/gen_baselines.py`, `data/references.json`.
Changed modules: `scoring.py` (D field, VERSION, docstring), `questions.py` (q11–q14), `store.py` (env-overridable path, safe append, `view()`, `training_grade()`, export fields), `app.py` (rewritten layout that reuses the existing helpers), `requirements.txt` (adds `hypothesis`, pinned).

## Components and Interfaces

### `diverge/scoring.py` (minimal change)

- Set `VERSION = "diverge-score-0.2"`. Extend the docstring with the D formula and an "experimental, NOT validated" note (Req 11.3).
- Inside `score_submission`, after the components are computed, add `result["divergence_score_d"] = round(clip(100*(0.50*div + 0.30*cnov + 0.20*chal), 0, 100), 1)`. If any input is missing or non-finite, the value is `None` (Req 11.1, 11.8). D is not gated and has no quality or correctness input (Req 11.2).
- Expose a pure helper `divergence_d(div, cnov, chal) -> float | None` so the formula can be property-tested.
- Legacy outputs `score`, `originality`, `quality`, `gate`, `components` and the others stay unchanged (Req 1.6). The signature also stays unchanged, which keeps `test_divergence_score_takes_no_correctness_input` passing.

### `diverge/config.py` (new)

```python
EVAL_CONFIG = dict(
    version="eval-0.1-experimental",
    scorer_version=scoring.VERSION, ivs_version=value.VERSION,
    ivs=value.DEFAULT_CONFIG,                       # weights, cap 25, floor 55, floor threshold 50
    entailment_threshold=0.7, borderline_margin=0.1, ivs_threshold=45.0,
    d_weights=dict(semantic_divergence=0.50, claim_novelty=0.30, assumption_challenge=0.20),
    scoring_constants=dict(weights=scoring.WEIGHTS, sim_same=scoring.SIM_SAME, sim_far=scoring.SIM_FAR,
                           claim_new_sim=scoring.CLAIM_NEW_SIM),
    models=dict(embedding="all-MiniLM-L6-v2", nli=models.NLI_NAME, lm=models.LLM_NAME),
    precision=dict(cpu="float32", mps="float16"), seed=7,
)
def canonical(cfg) -> str      # json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False)
def config_hash(cfg) -> str    # sha256(canonical(cfg).encode("utf-8")).hexdigest()
def validate(cfg) -> dict      # value.validate_config(cfg["ivs"]) + range checks; ValueError naming each bad field
def environment() -> dict      # device, precision, torch/transformers/sentence-transformers versions
```

### `diverge/correctness.py` (new)

```python
class Checker:
    def __init__(self, path=REFERENCES, question_ids=None): ...   # loads once; self.keys, self.warnings
    def is_verifiable(self, qid) -> bool
    def check(self, qid, answer, reasoning) -> dict
        # {status, method: "expected_answer"|"nli_claims"|"none",
        #  evidence: {...deciding...} | None, supplementary: [...], max_prob: float|None, note: str|None}
def extract_answer(kind, answer_text) -> float | str | None    # pure, model-free
def match_expected(expected, extracted) -> bool                  # pure, model-free
def validate_key(qid, key) -> list[str]                          # failed checks, [] if valid
```

- Loading (Req 2.1–2.5). A missing file or invalid JSON gives `keys={}` and one warning. Each key is checked with `validate_key`, and an invalid key adds a warning that names the question id and the failed check. Unknown question ids produce a warning and are ignored. The checker never raises during load.
- Expected_Answer path (Req 14.3–14.5, 14.7). This path runs first when a key has `expected_answer`, and it uses only the answer text.
  - `numeric`: strip `$ € £` and thousands separators, then take the first token matching `-?\d+(\.\d+)?(/\d+)?%?`. `a/b` is read as a fraction. `x%` matches if either `x` or `x/100` is within `tolerance`.
  - `string`: lowercase the text and remove punctuation. It matches if any accepted alias appears as a whole word and no known-wrong alias appears.
  - If nothing is extracted, the status is `unverified`. A match gives `verified_correct`, anything else gives `verified_incorrect`.
  - Deciding evidence is `{expected, extracted, citation}`. NLI claim results are still computed and stored as `supplementary`. If NLI fails, supplementary is `[]` and the decision stands.
- NLI-claims path (Req 3.1–3.7). Join the answer and reasoning, split them with `baselines.split_sentences`, and score every (claim, sentence) pair with `models.nli([(claim, sentence)])` (premise = claim). The checks run in this order:
  1. A Rejected_Claim entailed ≥ θ, or an Accepted_Claim contradicted ≥ θ, gives `verified_incorrect`.
  2. Otherwise, an Accepted_Claim entailed ≥ θ gives `verified_correct`.
  3. Otherwise the status is `unverified`.
  The deciding evidence is the best qualifying pair (claim, accepted/rejected, sentence, relation, prob rounded to 3 places, citation). For `unverified`, it is the best observed pair.
- No key gives `unverified` with note "no reference key" and no NLI call (Req 3.5, 5.7). An NLI exception gives `unverified` with note "verification unavailable" (Req 3.8).
- `check()` takes no baselines argument, so Req 3.9 and 14.7 hold by construction.
- `python -m diverge.correctness --check` prints load warnings and, for each open-ended key, the NLI contradiction of every Accepted_Claim against that question's cached Consensus_Assumptions. Authors use it to satisfy Req 2.7 when writing `references.json`.

### `diverge/evaluate.py` (new): the evaluation pipeline

```python
def evaluate(question_id, question, answer, reasoning, evidence, baselines, prior_texts,
             checker=None, config=EVAL_CONFIG) -> dict:
    cfg = config_mod.validate(config)                       # raises before any scoring (Req 4.11, 9.8)
    seed_everything(cfg["seed"])                            # random, numpy, torch (Req 9.6)
    full_reasoning = join(reasoning, evidence)              # evidence stored with reasoning (Req 16.7)
    ev = scoring.score_submission(question, answer, full_reasoning, baselines, prior_texts)
    try:   corr = (checker or default_checker()).check(question_id, answer, full_reasoning)
    except Exception as e: corr = unverified(note="verification unavailable"); warnings.append(...)  # Req 1.7, 12.5
    iv = value.intelligence_value(ev, corr["status"], cfg["ivs"])   # errors propagate -> caller stores nothing (Req 12.6)
    interval = uncertainty(question, answer, full_reasoning, baselines, prior_texts, ev, iv, corr, cfg)
    return dict(evaluation=ev, assessment=dict(
        divergence_score_d=ev["divergence_score_d"], dimensions=iv["dimensions"],
        correctness_status=corr["status"], verification=corr, ivs=iv["ivs"],
        merit_score=iv["merit_score"], novelty_credit=iv["novelty_credit"],
        breakdown=breakdown(iv, cfg), ivs_low=interval["low"], ivs_high=interval["high"],
        uncertainty_level=interval["level"], uncertainty_reasons=interval["reasons"],
        explanation=explain_v2(ev, iv, corr, interval, cfg),
        ivs_version=value.VERSION, config_version=cfg["version"], config_hash=config_hash(cfg),
        prior_pool_hashes=[sha256(t) for t in baselines + prior_texts], environment=environment(),
        warnings=warnings))
```

- `breakdown(iv, cfg)` (Req 6.1–6.4). It returns five entries: reasoning, evidence, correctness credit, semantic novelty and marginal value. Each entry has `raw`, `weight`, `gate` (novelty entries only) and `points` (2 dp). It also returns `pre_rule_ivs` and `rule: {name, threshold, before, after} | None`. Every number comes from `value.combine` outputs and config weights.
- `explain_v2` (Req 6.5–6.8, 14.6). It writes one sentence each for novelty, reasoning, correctness and marginal value, plus a gate sentence when gate < 1. The correctness sentence names the deciding claim, sentence, citation and probability, or the expected value, extracted value and citation, or "no valid reference key", or "no claim reached θ (best p)". The Challenge_Workspace shows the first four sentences (Req 16.8). The full list goes in the breakdown expander.
- `uncertainty(...)` (Req 7).
  - With N ≥ 2 baselines, it reruns `score_submission` N times with one baseline removed each time. The removed baseline drops out of the prior pool and the consensus assumptions automatically, because `score_submission` derives both from `baselines`. Each run is combined with `value.intelligence_value(ev_i, status)` using the same correctness status.
  - The low and high bounds are the min and max over these N + 1 values.
  - If the status is `unverified`, the bounds widen with `combine(dims, "verified_incorrect")` and `combine(dims, "verified_correct")`.
  - `level` comes from the width (< 10 low, 10–25 medium, > 25 high). It is forced to `high` with reason "borderline verification" when an NLI decision is within 0.1 of θ, or when an unverified verifiable item has a best probability ≥ θ − 0.1. Expected_Answer decisions skip the borderline check.
  - With fewer than 2 baselines, leave-one-out is skipped, bounds start at the IVS, and the reason "insufficient baselines" is added.
  - Cost: perplexity and NLI pairs are `lru_cache`d in `models.py`, so the extra runs mostly redo embeddings, which takes a few seconds on CPU.
- `assess_legacy(record) -> dict` gives an in-memory view for records without `assessment` (Req 8.6, 11.7, 13.5).
  - It computes IVS from the stored `components` using `unverified`, then sets `ivs_low`, `ivs_high` and `uncertainty_level` to `None` and `divergence_score_d` to `None`, which displays as "not computed".
  - It adds `not_rescored=True`, sets `breakdown=None` and `verification=None`. It never writes to disk.
  - Note: D is not derived for legacy records even though `details` holds the inputs, because Req 11.7 requires "not computed" until a record is rescored.
- `training_grade(ivs, status, consent, demo_seed, threshold) -> bool` (Req 13.1–13.2, 17.2–17.3). It returns `ivs is not None and ivs >= threshold and status != "verified_incorrect" and (consent or demo_seed)`. D and the legacy composite are not parameters.

### `diverge/store.py` (additive)

- `SUBMISSIONS = Path(os.environ.get("DIVERGE_SUBMISSIONS", ROOT/"data"/"submissions.jsonl"))`, so tests point at a temp copy.
- `add()` serializes first and appends one line with one `write()`. On exception it truncates back to the original size and re-raises, which leaves no partial record (Req 8.9).
- `view(record) -> dict` returns the record plus `a = record.get("assessment") or assess_legacy(record)`. Invalid statuses map to `unverified` with `status_flag="invalid status"`, and missing statuses use `status_flag="not rescored"` (Req 8.6, 8.8).
- `table()` gains the columns `d`, `ivs`, `ivs_low`, `ivs_high`, `uncertainty`, `status`, `status_flag`, `training_grade`, `consent`, `verifiable`. Existing columns stay.
- `leaderboard()` keeps its signature. `training_grade` now counts `table.training_grade` (Req 13.3). A new `impact_leaderboard(df)` sorts by verified-correct count, then training-grade count, then mean reasoning quality, all descending (Req 18.5).
- `to_training_record()` adds `correctness_status`, `verification`, `ivs`, `ivs_low`, `ivs_high`, `uncertainty_level`, `divergence_score_d` and `not_rescored` to the full record, or to `metadata` for the chat and DPO formats. The legacy `divergence_score` key still holds the legacy composite (Req 8.5, 8.7, 11.6).
- `TRAINING_THRESHOLD = 45.0` stays (the tests import it). IVS_Threshold reads `EVAL_CONFIG["ivs_threshold"]`.

### `diverge/questions.py` and offline baselines

Add q11–q14 with `verifiable=True`. q01–q10 stay unchanged.

| id | text (short) | Expected_Answer | known wrong | citation |
|---|---|---|---|---|
| q11 | Bat and ball cost $1.10 total; bat costs $1.00 more. How many cents is the ball? | numeric 5, tol 0 | 10 | Frederick (2005), J. Econ. Perspectives 19(4) |
| q12 | Monty Hall: probability of winning if you always switch? | numeric 2/3, tol 0.01 | 0.5 | Selvin (1975), The American Statistician 29(1) |
| q13 | Disease prevalence 1/1000, test sensitivity 100%, false-positive rate 5%: probability a positive has the disease? | numeric 0.0196, tol 0.002 | 0.95 | Casscells, Schoenberger & Graboys (1978), NEJM 299 |
| q14 | Smallest group size with > 50% chance two share a birthday (365 days)? | numeric 23, tol 0 | 183, 365 | Feller, An Introduction to Probability Theory, Vol. 1 |

q11 expects a value in cents. "$0.05" extracts as 0.05 and would be marked incorrect, so the q11 statement and the Req 14.10 hint both ask for the answer "in cents". This is noted as a limitation. Each of q11–q14 also gets one Accepted_Claim and one Rejected_Claim for supplementary NLI evidence.

Open-ended keys cover at least 3 questions from q01–q10. Candidates are q02 (induced demand; Duranton & Turner 2011, AER 101(6)), q05 (Grove et al. 2000, Psychological Assessment 12(1)) and q04 (efficacy vs. effectiveness; Rothwell 2005, Lancet 365). At least one Accepted_Claim is picked with `python -m diverge.correctness --check` so that it contradicts a cached consensus assumption at ≥ 0.7 (Req 2.7). Claims are written by the user or developer and are never generated by an LLM.

`diverge/gen_baselines.py` calls `get_baselines(BY_ID[qid]["text"])` for q11–q14 and is run once from the terminal (`.venv/bin/python -m diverge.gen_baselines`, about 20 s per question). The App always calls `get_baselines(..., generate_if_missing=False)` and shows "baselines not cached" if one is missing. Scoring still works with 0 or 1 baselines (Req 14.8–14.9). With 0 baselines, `score_submission` cannot embed an empty list, so `evaluate()` guards that case: D is `None`, and the novelty dimensions come from an empty pool. This is covered by an edge-case test.

## Data Models

### Stored record (one JSONL line, additive)

Existing keys are unchanged: `id, user, question_id, domain, question, answer, reasoning, baselines, baseline_meta, evaluation, demo_seed, submitted_at`. `evaluation` now also contains `divergence_score_d`, and `meta.version` is `diverge-score-0.2`.

New keys:

```json
{
  "evidence_text": "string (also appended to reasoning)",
  "training_consent": false,
  "assessment": {
    "divergence_score_d": 61.3,
    "dimensions": {"semantic_novelty": 0, "reasoning_quality": 0, "evidence_quality": 0, "marginal_value": 0},
    "correctness_status": "verified_correct | verified_incorrect | unverified",
    "verification": {"method": "expected_answer | nli_claims | none", "evidence": {}, "supplementary": [], "max_prob": 0.0, "note": null},
    "ivs": 58.2, "merit_score": 0, "novelty_credit": 0,
    "breakdown": {"entries": [{"name": "reasoning_quality", "raw": 0, "weight": 0.3, "gate": null, "points": 0}],
                  "pre_rule_ivs": 0, "rule": null},
    "ivs_low": 51.0, "ivs_high": 63.4, "uncertainty_level": "medium", "uncertainty_reasons": [],
    "explanation": ["..."],
    "ivs_version": "ivs-0.1-experimental", "config_version": "eval-0.1-experimental",
    "config_hash": "sha256…", "prior_pool_hashes": ["sha256…"], "environment": {"device": "cpu"},
    "warnings": []
  }
}
```

Legacy records have no `assessment`, `training_consent` or `evidence_text`. `store.view()` supplies `assess_legacy()` in memory, with consent treated as false.

### `data/references.json`

```json
{
  "q12": {
    "accepted_claims": [{"text": "Switching wins with probability 2/3.", "citation": "Selvin (1975) ..."}],
    "rejected_claims": [{"text": "After one door is opened the two remaining doors are equally likely.", "citation": "..."}],
    "expected_answer": {"kind": "numeric", "value": 0.6667, "tolerance": 0.01,
                        "known_wrong": [0.5], "citation": "Selvin (1975) ..."}
  }
}
```

For `kind: "string"` the shape is `"accepted": [1..10 aliases], "known_wrong": [aliases]`.

## UI Design (`app.py`)

- Router: `st.session_state.page ∈ {HOME, CHALLENGES, MY IMPACT, ABOUT}`, drawn by a top nav of four `st.button`s in `st.columns` (Req 15.1). `st.session_state.challenge` holds the open question id. CSS is the existing block with the background switched to charcoal `#0B0A10`. `card`, `style`, `gauge`, `radar` and `contributions_bar` are reused.
- The page reads `recs = store.load_all()` and `df = store.table(recs)` at the top of every rerun, so MY IMPACT always reflects the latest store (Req 18.6).
- Status text labels come from one helper, `badge(status, flag)`, which renders "Verified correct", "Verified incorrect" or "Unverified" plus an optional "not rescored" or "invalid status". The label is the main cue and color is secondary (Req 8.1, 20.4).

HOME
- Hero copy as specified, with "Start a Challenge" setting `page="CHALLENGES"` and "How It Works" opening the steps expander (Req 15.3–15.5).
- Four step cards and a climate chain labeled "hypothesis" (Req 15.6–15.7).
- Up to 4 values read from `X.summarize`/`X.significance` with provenance (file, `completed_at`, seed count), or "pending" (Req 15.8–15.9). No tables (Req 15.10).

CHALLENGES
- List view: one row per question with title, difficulty, domain, Verification_Type ("Objective answer" if verifiable and the checker holds an Expected_Answer, else "Open-ended") and Solved for the session Display_Name (Req 16.1–16.2). An "Open" button sets `challenge`.
- Workspace, two `st.columns` (Req 16.3):
  - Left column: title and statement, the first cached baseline under "AI baseline approach", consensus assumptions under "assumptions to investigate (hypotheses, not errors)", every baseline in an expander, and the Consensus Breaker as an expander that keeps today's assumption-challenge rendering and custom-question option (Req 15.2, 16.4).
  - Right column: inputs for Display_Name, answer, reasoning and "evidence or proposed verification test", a consent checkbox (default False), the verifiable hint, and "Submit for Evaluation" (Req 16.5, 16.12, 14.10).
  - Validation (name 1–40 characters with no "@", non-empty answer) shows a field-specific `st.error`, and the widget keys keep the entered text (Req 16.6).
  - On submit, `evaluate()` runs and then `store.add()`. A store error shows "not saved" and the inputs are kept (Req 8.9). An IVS error shows D and the legacy composite plus an error, and nothing is stored (Req 12.6).
- Result view, every value read from the `assessment` or `evaluation` dict with no maths in the UI (Req 16.9):
  - Three cards: "Divergence Score (originality only, not a quality measure)" (D), "Reasoning Quality (estimate)" (Q), and "Estimated Learning Value (estimate)" (IVS) with low/high and Uncertainty_Level. The badge is shown, plus "correctness not checked against a human reference" when the status is unverified, or "capped at 25" when it is incorrect.
  - Up to 4 explanation sentences.
  - An expander with the breakdown table, the cap/floor record, all sentences, and Marginal_Value labeled "estimated proxy, not measured learning gain". The legacy gauge and radar are labeled "legacy composite" (Req 6.9, 11.4–11.5, 12.1–12.4, 12.7, 17.5).
- A "How scoring works" expander with the Req 16.10–16.11 copy.

MY IMPACT
- Display_Name input, defaulting from the session. Metrics per Req 18.1–18.2: average D labeled "how different, not how good", average IVS labeled "Estimated Learning Value (estimate)", and "no data" when empty.
- A "Your contribution to greener AI" pathway. The first two steps show the user's counts (submissions, training-grade). The last three show "not yet measured" plus a statement that energy cannot be attributed per user. No per-user kWh is shown (Req 18.3–18.4).
- `store.impact_leaderboard` with the demo label (Req 18.5).
- Export_Panel expander: the current Training Data view plus three status toggles (correct and unverified on, incorrect off), a minimum IVS slider defaulting to 45, the minimum gate, demo seeds, a consent note, 0 selected → disabled button, and export of selected ids only (Req 8.3–8.4, 13.4, 19.10).

ABOUT
- Vision, Mission, Goal and Belief verbatim, plus methodology prose (Req 19.1–19.2).
- Three bordered blocks:
  - "Measured performance": today's experiment charts and tables from `X.*`, the config, seeds, dataset, model and run date, plus the null-result sentence. If results are missing it shows "pending" and the run command (Req 19.4–19.5).
  - "Estimated energy impact": only analytic FLOPs and wall-clock from `results.json` (Req 19.9).
  - "Hypothetical scenarios": the Energy_Calculator. Inputs are power in kW and duration in hours, each labeled "assumption", and the reduction defaults to 0%. Baseline kWh = P × h, scenario h = h × (1 − r), savings = baseline − scenario. The block includes the rebound, utilization and inference caveat (Req 19.6–19.8).
- The Method table, updated with D and IVS.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Pure properties run model-free with stubbed component values.

### Property 1: Novelty is a bonus, never a penalty
For any reasoning, evidence and correctness status, and any two (semantic_novelty, marginal_value) pairs where A ≥ B componentwise, `combine(A).ivs ≥ combine(B).ivs`.
**Validates: Requirements 4.7**

### Property 2: Incorrect answers are capped and never training-grade
For any dimensions in 0–100, any Divergence_Score_D in 0–100 and any consent or demo flag, a `verified_incorrect` result has `ivs ≤ incorrect_cap`, and `training_grade(...)` is false.
**Validates: Requirements 4.4, 4.8, 10.10, 13.1, 17.3, 5.6**

### Property 3: Correct conventional answers reach the floor
For any semantic_novelty, marginal_value and evidence in 0–100 (including all 0) and reasoning ≥ floor threshold, `combine(..., "verified_correct").ivs ≥ conventional_floor`.
**Validates: Requirements 4.5, 4.9, 5.3, 17.4**

### Property 4: Correctness ordering
For any fixed dimensions, IVS(verified_correct) ≥ IVS(unverified) ≥ IVS(verified_incorrect).
**Validates: Requirements 4.10, 5.2**

### Property 5: Separation of signals
For any dimensions, `merit_score` is independent of semantic_novelty and marginal_value, and `novelty_credit` is independent of correctness status. For any (div, cnov, chal), `divergence_d` stays the same whatever reasoning, evidence, gate or status is used.
**Validates: Requirements 1.4, 1.5, 4.1, 4.2, 5.1, 10.11, 11.2, 17.1**

### Property 6: Ranges and validation
For any valid inputs, ivs, merit_score, novelty_credit and divergence_d are in 0–100 and rounded as specified. For any out-of-range, non-finite, non-numeric or bool dimension, any unknown status, or any config breaking a weight-sum, range or cap<floor constraint, a `ValueError` naming the field is raised.
**Validates: Requirements 1.1, 4.3, 4.6, 4.11, 4.12, 9.8, 11.1**

### Property 7: Interval contains the score
For any dimensions, status and set of leave-one-out dimension vectors, the interval built by `evaluate.interval_from(...)` satisfies `low ≤ ivs ≤ high` in 0–100. The level follows the width rule unless a reason forces `high`. "insufficient baselines" is present if and only if there are fewer than 2 baselines.
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.7, 7.8**

### Property 8: Breakdown sums to the pre-rule IVS
For any inputs, the breakdown entry points sum to `pre_rule_ivs` within ±0.1. `rule` is non-null exactly when a cap or floor changed the value, and then `before ≠ after`.
**Validates: Requirements 6.1, 6.2, 6.3, 6.4**

### Property 9: Config hash round trip
For any generated Evaluation_Config, `parse(canonical(cfg)) == cfg` and the hash is equal. Changing any single leaf value changes the hash.
**Validates: Requirements 9.7**

### Property 10: Metadata does not affect scores
For any author name (0–100 Unicode characters), timestamp (2000–2100) and demo flag, `evaluate()` with stubbed `models` returns identical score fields, excluding timing metadata.
**Validates: Requirements 5.4, 5.5**

### Property 11: Expected-answer matching
For any numeric Expected_Answer (value, tolerance) and any number x, an answer text stating x first (optionally with `$`, a thousands separator, `%` or a/b form) gives `verified_correct` if and only if |x − value| ≤ tolerance. Text with no number gives `unverified`. For any string key, the text contains an accepted alias and no wrong alias if and only if the result is `verified_correct`.
**Validates: Requirements 14.3, 14.4, 10.12**

### Property 12: Correctness ignores baselines
For any submission text and reference key, `Checker.check` (with stubbed NLI) returns the same status and evidence regardless of baselines. This holds structurally, because baselines are not a parameter, and the test also calls `evaluate()` with randomized baseline lists and a stubbed scorer.
**Validates: Requirements 3.9, 5.1, 14.7**

### Property 13: Reference-key validation
For any generated key, `validate_key` returns [] if and only if all Req 2.2 constraints hold. Loading a file with a mix of valid and invalid keys keeps every valid key and warns once per invalid or unknown id.
**Validates: Requirements 2.2, 2.3, 2.5**

### Property 14: Export round trip and legacy safety
For any generated assessed or legacy record written to a temp store, exporting in each of the 3 formats and parsing the result gives the stored status, verification, ivs, ivs_low and ivs_high (1 dp). Legacy records export `unverified` and `not_rescored=true`. The source file bytes stay unchanged.
**Validates: Requirements 8.2, 8.5, 8.6, 8.7, 8.10, 11.6, 11.7, 13.5**

## Error Handling

| Failure | Behavior |
|---|---|
| `references.json` missing or invalid JSON | All questions `unverified`. One load warning is shown in the "How scoring works" expander. No exception (Req 2.4). |
| Invalid or unknown key | Key skipped, warning names the id and check (Req 2.3, 2.5). |
| Checker raises or NLI unavailable | Status `unverified`, note "verification unavailable", warning in `assessment.warnings` and a UI `st.warning`. The record is still stored (Req 1.7, 3.8, 12.5). |
| `value.combine` raises | UI shows D and the legacy composite plus an error. Nothing is stored (Req 12.6). |
| Invalid Evaluation_Config | `evaluate()` raises before scoring and the store is untouched (Req 4.11, 9.8). |
| Store write fails | File truncated to its original size, UI says "not saved", inputs kept (Req 8.9). |
| Baselines not cached | Message shown with the generator command. Scoring proceeds with 0–1 baselines and "insufficient baselines" (Req 14.9). |
| Missing `results.json` | "pending" on HOME and ABOUT (Req 15.9, 19.5). |
| Legacy or invalid-status record | In-memory view flagged "not rescored" or "invalid status". The file is never rewritten (Req 8.6, 8.8, 13.6). |

## Testing Strategy

- Run everything with `.venv/bin/python -m pytest`. Add `hypothesis` (pinned, Python 3.9-compatible, e.g. `hypothesis==6.122.3`, version confirmed at install) to `requirements.txt`. Use a profile in `tests/conftest.py` with `settings.register_profile("diverge", max_examples=100, derandomize=True)`, which satisfies Req 10.1. The existing 24 tests in `tests/test_scoring.py` stay unchanged and must keep passing.
- Property tests go in `tests/test_properties.py` and are model-free, staying under 60 s (Req 10.2–10.4, 10.10–10.12). There is one Hypothesis test per property above, each tagged with a comment `# Feature: bias-resistant-divergence-scoring, Property N: <title>`. Property 10 and Property 12 monkeypatch `models.embed/nli/perplexity` with deterministic stubs.
- Model fixture tests go in `tests/test_correctness_models.py`, marked `models`, and skip with the model name if the local cache is missing (Req 10.5–10.8, 2.8):
  - (a) wrong contrarian answer on a keyed question is `verified_incorrect` with IVS ≤ 25
  - (b) a correct paraphrase with cos ≥ 0.85 is `verified_correct` with IVS ≥ 55
  - (c) an answer that contradicts the consensus and is correct is `verified_correct`
  - (d) 30 seeded random words give gate 0 and novelty_credit 0
  - edge cases: empty reasoning, single sentence, no key, missing file, malformed key, fewer than 2 baselines
  - a determinism test that scores twice
  - the Req 2.7 contradiction check against shipped files
  - `demo:noise` and `demo:offtopic` are not training-grade (Req 10.14, 17.7)
- Legacy tests copy `data/submissions.jsonl` to `tmp_path` and set `DIVERGE_SUBMISSIONS`. They check that all 11 records load through `store.view`, that `table` and `leaderboard` build, that all 3 export formats parse, and that the sha256 of the copy and the real file are unchanged (Req 10.13, 13.6).
- App smoke tests go in `tests/test_app.py` and use `streamlit.testing.v1.AppTest.from_file("app.py")`. An autouse fixture copies `submissions.jsonl` to `tmp_path` and sets `DIVERGE_SUBMISSIONS`, so real data is never touched.
  - For each of the four pages, the test clicks the nav button, asserts no `at.exception`, and checks key copy: hero headline, Verification_Type column, "not yet measured", and the "assumption" labels with default 0 kWh savings.
  - Submission flow (marked `models`): open q12, enter a name and answer "2/3 because…", tick consent, submit, then assert the "Verified correct" badge, the D and IVS labels, and that the temp file has 12 lines while the real file hash is unchanged.
  - A validation case: a name with "@" gives an error, and the file line count is unchanged.
- Manual checks after implementation: the running :8501 server hot-reloads without errors, and `.venv/bin/python -m diverge.seed` still scores without errors. Do not start a second server.
