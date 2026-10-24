"""
Intelligence Value Score (IVS), EXPERIMENTAL (weights chosen by hand, NOT validated).

Additive layer on top of `scoring.score_submission` (diverge-score-0.1), following
Requirement 4 of .kiro/specs/bias-resistant-divergence-scoring. The legacy
Divergence Score is left unchanged; IVS is reported alongside it.

    Merit_Score    = w_reasoning * Reasoning_Quality + w_evidence * Evidence_Quality
                   + w_correctness * Correctness_Credit            (no novelty input)
    Novelty_Credit = Quality_Gate * (w_novelty * Semantic_Novelty + w_marginal * Marginal_Value)
    IVS            = Merit_Score + Novelty_Credit
                     -> min(IVS, Incorrect_Cap)       if verified_incorrect
                     -> max(IVS, Conventional_Floor)  if verified_correct and Reasoning_Quality >= threshold
                     -> clip to 0..100, round to 1 decimal

Correctness_Status is an INPUT here. It must come from a human-authored
reference (planned `data/references.json` + correctness checker), never from the
AI baselines or any LLM. Until that checker exists, callers pass `unverified`
or a human-assigned label. Marginal_Value is an estimated proxy, not a measured
learning gain.
"""
from __future__ import annotations

import math

VERSION = "ivs-0.1-experimental"

STATUSES = ("verified_correct", "unverified", "verified_incorrect")
CORRECTNESS_CREDIT = {"verified_correct": 100.0, "unverified": 50.0, "verified_incorrect": 0.0}

DEFAULT_CONFIG = dict(
    w_reasoning=0.30, w_evidence=0.15, w_correctness=0.30, w_novelty=0.15, w_marginal=0.10,
    incorrect_cap=25.0, conventional_floor=55.0, floor_quality_threshold=50.0,
)
_WEIGHT_KEYS = ("w_reasoning", "w_evidence", "w_correctness", "w_novelty", "w_marginal")


def validate_config(cfg: dict) -> dict:
    """Raise ValueError naming the violated constraint (Requirement 4.6 / 4.11)."""
    for k in _WEIGHT_KEYS:
        v = cfg[k]
        if not (isinstance(v, (int, float)) and 0.0 <= v <= 1.0):
            raise ValueError(f"{k} must be in 0-1, got {v!r}")
    total = sum(cfg[k] for k in _WEIGHT_KEYS)
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"weights must sum to 1.0 (+-0.001), got {total!r}")
    for k in ("incorrect_cap", "conventional_floor", "floor_quality_threshold"):
        if not (0.0 <= cfg[k] <= 100.0):
            raise ValueError(f"{k} must be in 0-100, got {cfg[k]!r}")
    if not cfg["incorrect_cap"] < cfg["conventional_floor"]:
        raise ValueError(f"incorrect_cap ({cfg['incorrect_cap']!r}) must be below "
                         f"conventional_floor ({cfg['conventional_floor']!r})")
    return cfg


def _check_dim(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 100.0:
        raise ValueError(f"{name} must be a finite number in 0-100, got {v!r}")
    return float(v)


def quality_gate(reasoning_quality: float) -> float:
    """Same gate as scoring.score_submission, on the 0-100 scale."""
    return min(1.0, max(0.0, (reasoning_quality / 100.0 - 0.25) / 0.35))


def combine(semantic_novelty, reasoning_quality, evidence_quality, marginal_value,
            correctness_status: str, config: dict | None = None) -> dict:
    """Pure score combination from 0-100 dimensions (Requirement 4)."""
    cfg = validate_config(dict(DEFAULT_CONFIG, **(config or {})))
    sn = _check_dim("semantic_novelty", semantic_novelty)
    rq = _check_dim("reasoning_quality", reasoning_quality)
    eq = _check_dim("evidence_quality", evidence_quality)
    mv = _check_dim("marginal_value", marginal_value)
    if correctness_status not in STATUSES:
        raise ValueError(f"correctness_status must be one of {STATUSES}, got {correctness_status!r}")

    gate = quality_gate(rq)
    merit = cfg["w_reasoning"] * rq + cfg["w_evidence"] * eq + cfg["w_correctness"] * CORRECTNESS_CREDIT[correctness_status]
    novelty_credit = gate * (cfg["w_novelty"] * sn + cfg["w_marginal"] * mv)
    pre = merit + novelty_credit

    rule, ivs = None, pre
    if correctness_status == "verified_incorrect" and pre > cfg["incorrect_cap"]:
        rule, ivs = "incorrect_cap", cfg["incorrect_cap"]
    elif (correctness_status == "verified_correct" and round(rq, 1) >= cfg["floor_quality_threshold"]
          and pre < cfg["conventional_floor"]):
        rule, ivs = "conventional_floor", cfg["conventional_floor"]
    ivs = round(min(100.0, max(0.0, ivs)), 1)

    return dict(ivs=ivs, merit_score=round(merit, 2), novelty_credit=round(novelty_credit, 2),
                quality_gate=round(gate, 3), pre_rule_ivs=round(pre, 2), rule_applied=rule,
                correctness_status=correctness_status, version=VERSION, config=cfg)


def dimensions(ev: dict) -> dict:
    """Map a `score_submission` result to the 0-100 dimensions (Requirement 1.2 / 1.3)."""
    c = ev["components"]
    clip = lambda x: round(min(100.0, max(0.0, 100.0 * x)), 1)  # noqa: E731
    return dict(semantic_novelty=clip(0.55 * c["semantic_divergence"] + 0.45 * c["conceptual_novelty"]),
                reasoning_quality=clip(c["reasoning_quality"]),
                evidence_quality=clip(c["evidence_quality"]),
                marginal_value=clip(c["information_gain"]))


def intelligence_value(ev: dict, correctness_status: str = "unverified", config: dict | None = None) -> dict:
    """IVS for an existing `score_submission` result. Does not modify `ev`."""
    d = dimensions(ev)
    return dict(dimensions=d, **combine(correctness_status=correctness_status, config=config, **d))
