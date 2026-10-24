"""
Scoring tests for DIVERGE.

Two scores are under test:
  * Divergence Score  - legacy `diverge.scoring.score_submission` (diverge-score-0.1), unchanged.
  * Intelligence Value Score (IVS) - `diverge.value`, an additive, correctness-aware layer.

Correctness labels below are assigned by the test author (human judgement), not
derived from the AI baselines. The automatic correctness checker from the
bias-resistant-divergence-scoring spec does not exist yet.

Two layers:
  * pure tests on `value.combine` over a grid of 0-100 inputs (fast, no models)
  * integration tests that run the real local models on cached baselines (marked `models`)
"""
from __future__ import annotations

import inspect
import itertools

import pytest

from diverge import store, value
from diverge.scoring import score_submission

GRID = (0.0, 25.0, 49.9, 50.0, 75.0, 100.0)
CFG = value.DEFAULT_CONFIG
THRESHOLD = store.TRAINING_THRESHOLD  # 45: the "training-grade" cut-off used by the app


def grid():
    for sn, rq, eq, mv in itertools.product(GRID, repeat=4):
        yield dict(semantic_novelty=sn, reasoning_quality=rq, evidence_quality=eq, marginal_value=mv)


# ---------------------------------------------------------------- real-model fixtures
Q02 = "q02"  # "Why do cities that build more roads often end up with worse traffic ..."

# Fluent, evidence-shaped, far from the AI baselines, and wrong (the numbers and mechanism are invented).
WRONG_CONTRARIAN = (
    "Traffic gets worse after road building because asphalt absorbs sunlight and heats the air, and hot air "
    "makes drivers slower. The fix is to paint every road white.",
    "Studies show that dark pavement raises local air temperature by 3 degrees, which therefore leads to slower "
    "reaction times. For example, in Phoenix commuters drive 12% slower on hot days. We should predict that "
    "white roads cut congestion by 30%.",
)
# Close to the AI baselines and consistent with the transport-economics literature (induced demand).
CORRECT_CONVENTIONAL = (
    "Building more roads often makes traffic worse because of induced demand: new capacity makes driving easier, "
    "so more people drive until the roads are congested again.",
    "Because the extra lanes lower the time cost of each trip, people take trips they used to skip and shift from "
    "transit to cars. As a result congestion returns. Reducing congestion therefore needs better public transport, "
    "congestion pricing and land use that shortens trips.",
)


@pytest.fixture(scope="module")
def real():
    """Score the fixtures once with the real local models and cached baselines."""
    try:
        from diverge import models
        from diverge.baselines import get_baselines
        from diverge.questions import BY_ID
        models.embed(["warm"])
    except Exception as e:  # pragma: no cover - environment without cached models
        pytest.skip(f"local models unavailable: {e}")

    def baselines(qid):
        b = get_baselines(BY_ID[qid]["text"], generate_if_missing=False)
        if b is None:
            pytest.skip(f"no cached baselines for {qid}")
        return b["baselines"]

    q = BY_ID[Q02]["text"]
    b02 = baselines(Q02)
    out = dict(q=q, b02=b02, b05=baselines("q05"))
    out["wrong"] = score_submission(q, *WRONG_CONTRARIAN, b02, [])
    out["correct"] = score_submission(q, *CORRECT_CONVENTIONAL, b02, [])
    # Same answer, scored against a different (unrelated) set of AI baselines.
    out["correct_other_baselines"] = score_submission(q, *CORRECT_CONVENTIONAL, out["b05"], [])
    return out


# ================================================================ 1. independence
class TestIndependence:
    def test_divergence_score_takes_no_correctness_input(self):
        params = set(inspect.signature(score_submission).parameters)
        assert params == {"question", "answer", "reasoning", "baselines", "prior_texts"}

    def test_merit_has_no_novelty_input(self):
        """Merit_Score must not move when only Semantic_Novelty / Marginal_Value change."""
        for status in value.STATUSES:
            for rq, eq in itertools.product(GRID, repeat=2):
                merits = {value.combine(sn, rq, eq, mv, status)["merit_score"]
                          for sn, mv in itertools.product(GRID, repeat=2)}
                assert len(merits) == 1, (status, rq, eq, merits)

    def test_novelty_credit_has_no_correctness_input(self):
        for d in grid():
            credits = {value.combine(**d, correctness_status=s)["novelty_credit"] for s in value.STATUSES}
            assert len(credits) == 1, d

    def test_novelty_is_a_bonus_never_a_penalty(self):
        for status in value.STATUSES:
            for rq, eq in itertools.product(GRID, repeat=2):
                for (sn_a, mv_a), (sn_b, mv_b) in itertools.product(itertools.product(GRID, repeat=2), repeat=2):
                    if sn_a >= sn_b and mv_a >= mv_b:
                        a = value.combine(sn_a, rq, eq, mv_a, status)["ivs"]
                        b = value.combine(sn_b, rq, eq, mv_b, status)["ivs"]
                        assert a >= b

    @pytest.mark.models
    def test_correctness_label_changes_ivs_but_not_divergence_score(self, real):
        import copy
        ev = real["correct"]
        before = copy.deepcopy({k: ev[k] for k in ("score", "originality", "quality", "gate", "components")})
        ivs = {s: value.intelligence_value(ev, s)["ivs"] for s in value.STATUSES}
        assert ivs["verified_correct"] > ivs["unverified"] > ivs["verified_incorrect"]
        # The legacy Divergence Score fields are identical after every correctness label was applied.
        assert {k: ev[k] for k in before} == before
        assert "ivs" not in ev

    @pytest.mark.models
    def test_swapping_ai_baselines_moves_divergence_but_not_merit(self, real):
        a, b = real["correct"], real["correct_other_baselines"]
        assert a["score"] != b["score"]
        assert a["components"]["semantic_divergence"] != b["components"]["semantic_divergence"]
        ma = value.intelligence_value(a, "verified_correct")["merit_score"]
        mb = value.intelligence_value(b, "verified_correct")["merit_score"]
        assert ma == pytest.approx(mb, abs=0.1)

    @pytest.mark.models
    def test_scores_can_rank_submissions_in_opposite_order(self, real):
        w, c = real["wrong"], real["correct"]
        assert w["score"] > c["score"], "wrong contrarian answer should be more divergent"
        assert (value.intelligence_value(c, "verified_correct")["ivs"]
                > value.intelligence_value(w, "verified_incorrect")["ivs"])


# ================================================================ 2. unconventional but incorrect
class TestUnconventionalIncorrect:
    def test_incorrect_is_capped_for_every_input(self):
        for d in grid():
            r = value.combine(**d, correctness_status="verified_incorrect")
            assert r["ivs"] <= CFG["incorrect_cap"] < THRESHOLD, d

    def test_maximal_novelty_and_quality_still_capped(self):
        r = value.combine(100, 100, 100, 100, "verified_incorrect")
        assert r["ivs"] == CFG["incorrect_cap"] and r["rule_applied"] == "incorrect_cap"

    @pytest.mark.models
    def test_fluent_wrong_contrarian_answer_is_not_training_grade(self, real):
        ev = real["wrong"]
        # Sanity: the fixture really is unconventional and passes the legacy quality gate.
        assert ev["details"]["semantic"]["max_baseline_similarity"] < 0.75
        assert ev["gate"] > 0.9
        iv = value.intelligence_value(ev, "verified_incorrect")
        assert iv["ivs"] <= CFG["incorrect_cap"]
        assert iv["ivs"] < THRESHOLD


# ================================================================ 3. conventional but correct
class TestConventionalCorrect:
    def test_correct_with_adequate_reasoning_hits_floor_even_with_zero_novelty(self):
        for rq in (50.0, 75.0, 100.0):
            for eq in GRID:
                r = value.combine(0.0, rq, eq, 0.0, "verified_correct")
                assert r["ivs"] >= CFG["conventional_floor"] > THRESHOLD, (rq, eq)

    def test_floor_needs_adequate_reasoning(self):
        r = value.combine(0.0, 49.9, 0.0, 0.0, "verified_correct")
        assert r["rule_applied"] is None and r["ivs"] < CFG["conventional_floor"]

    @pytest.mark.models
    def test_correct_paraphrase_of_ai_consensus_is_training_grade(self, real):
        ev = real["correct"]
        # Sanity: the fixture really is conventional (close to an AI baseline).
        assert ev["details"]["semantic"]["max_baseline_similarity"] > 0.75
        iv = value.intelligence_value(ev, "verified_correct")
        assert iv["dimensions"]["reasoning_quality"] >= CFG["floor_quality_threshold"]
        assert iv["ivs"] >= CFG["conventional_floor"]
        assert iv["ivs"] >= THRESHOLD


# ================================================================ 4. ranges
class TestRanges:
    def test_ivs_and_parts_in_range_for_every_input(self):
        for d in grid():
            for s in value.STATUSES:
                r = value.combine(**d, correctness_status=s)
                assert 0.0 <= r["ivs"] <= 100.0
                assert 0.0 <= r["merit_score"] <= 100.0
                assert 0.0 <= r["novelty_credit"] <= 100.0

    @pytest.mark.parametrize("bad", [-0.1, 100.1, float("nan"), float("inf"), "50", None, True])
    def test_out_of_range_input_is_rejected(self, bad):
        with pytest.raises(ValueError, match="semantic_novelty"):
            value.combine(bad, 50, 50, 50, "unverified")

    def test_unknown_status_is_rejected(self):
        with pytest.raises(ValueError, match="correctness_status"):
            value.combine(50, 50, 50, 50, "probably_correct")

    def test_invalid_config_is_rejected(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            value.combine(50, 50, 50, 50, "unverified", config=dict(w_reasoning=0.9))
        with pytest.raises(ValueError, match="incorrect_cap"):
            value.combine(50, 50, 50, 50, "unverified", config=dict(incorrect_cap=60.0))

    @pytest.mark.models
    def test_real_scores_in_range_including_noise_and_off_topic(self, real):
        from diverge.questions import BY_ID
        from diverge.seed import SEEDS
        from diverge.baselines import get_baselines

        evs = [real["wrong"], real["correct"], real["correct_other_baselines"]]
        for _user, qid, ans, why in SEEDS:  # includes gibberish, off-topic and paraphrase demo entries
            q = BY_ID[qid]["text"]
            b = get_baselines(q, generate_if_missing=False)
            if b is not None:
                evs.append(score_submission(q, ans, why, b["baselines"], []))
        assert len(evs) > 3
        for ev in evs:
            for k in ("score", "originality", "quality"):
                assert 0.0 <= ev[k] <= 100.0, (k, ev[k])
            assert all(0.0 <= v <= 1.0 for v in ev["components"].values())
            iv = value.intelligence_value(ev, "unverified")
            assert all(0.0 <= v <= 100.0 for v in iv["dimensions"].values())
            for s in value.STATUSES:
                assert 0.0 <= value.intelligence_value(ev, s)["ivs"] <= 100.0
