"""
DIVERGE Score v0.1 (experimental weights, NOT validated).

    score = 100 * ( 0.30 * semantic_divergence * gate
                  + 0.25 * conceptual_novelty  * gate
                  + 0.25 * reasoning_quality
                  + 0.10 * evidence_quality
                  + 0.10 * information_gain    * gate )

Originality (divergence, novelty, information gain) is computed separately from
quality (reasoning, evidence). Originality is only *credited* in proportion to a
quality gate, so off-topic, incoherent or random text cannot score high just
because it is far from the AI baselines.

Component definitions
---------------------
semantic_divergence  1 - max cosine(answer, AI baseline), calibrated to [0,1]
conceptual_novelty   0.6 * claim novelty (share of the answer's claims with no
                     close counterpart in baselines or prior submissions)
                     + 0.4 * assumption challenge (NLI contradiction of the
                     consensus assumptions shared across baselines)
reasoning_quality    relevance to question, LLM fluency (perplexity), internal
                     consistency (NLI between the answer's own sentences),
                     explicit causal structure, and depth (length) adequacy
evidence_quality     transparent heuristics: quantities, named examples,
                     mechanisms, sources, testable predictions
information_gain     residual variance of the answer embedding after projecting
                     onto the span of all existing answers (baselines + prior
                     submissions) = log-det (DPP) volume gain, in [0,1]
"""
from __future__ import annotations

import math
import re
import time

import numpy as np

from . import models
from .baselines import consensus_assumptions, split_sentences

VERSION = "diverge-score-0.1"
WEIGHTS = dict(semantic_divergence=0.30, conceptual_novelty=0.25, reasoning_quality=0.25,
               evidence_quality=0.10, information_gain=0.10)
GATED = ("semantic_divergence", "conceptual_novelty", "information_gain")

# Calibration anchors for all-MiniLM-L6-v2 cosine similarity (chosen by inspection,
# not fitted): ~0.85 = paraphrase, ~0.25 = unrelated text on a different topic.
SIM_SAME, SIM_FAR = 0.85, 0.25
CLAIM_NEW_SIM = 0.55  # a claim is "new" if its best match in existing text is below this

CAUSAL = r"\b(because|therefore|thus|hence|so that|which means|leads? to|causes?|results? in|due to|" \
         r"implies|if\b.*\bthen|unless|whereas|however|but|instead|rather than|as a result|in turn)\b"
EVIDENCE_PATTERNS = {
    "quantities": r"\b\d+(\.\d+)?\s?(%|percent|x|times|kwh|mwh|gw|mw|tons?|years?|hours?|\$|usd|billion|million)?\b",
    "named examples": r"\b(for example|for instance|e\.g\.|such as|case of|in [A-Z][a-z]+|[A-Z][a-z]+ (study|paper|report))\b",
    "mechanism": r"\b(mechanism|feedback|incentive|bottleneck|constraint|trade-?off|equilibrium|externalit|selection effect)\w*",
    "sources": r"(https?://|doi\.org|\bet al\.|\(\d{4}\)|\bstudy\b|\bstudies\b|\bdata\b|\bevidence\b|\btrial\b|\bsurvey\b)",
    "testable prediction": r"\b(predict|would expect|we should see|testable|measur|experiment|falsif|counterfactual)\w*",
}


def _cal(sim: float) -> float:
    return float(np.clip((SIM_SAME - sim) / (SIM_SAME - SIM_FAR), 0, 1))


def _sat(x: float, k: float) -> float:
    return float(1 - math.exp(-x / k))


# ---------------------------------------------------------------- components
def semantic_divergence(ans_vec, base_vecs):
    sims = base_vecs @ ans_vec
    return _cal(float(sims.max())), dict(max_baseline_similarity=round(float(sims.max()), 3),
                                         mean_baseline_similarity=round(float(sims.mean()), 3))


def claim_novelty(claims, existing_sents):
    if not claims:
        return 0.0, []
    C = models.embed(claims)
    E = models.embed(existing_sents) if existing_sents else np.zeros((0, C.shape[1]))
    best = (C @ E.T).max(axis=1) if len(E) else np.zeros(len(claims))
    new = [dict(claim=c, nearest_similarity=round(float(s), 3)) for c, s in zip(claims, best)]
    score = float(np.mean([_cal(s) for s in best]))
    return score, sorted(new, key=lambda r: r["nearest_similarity"])


def assumption_challenge(assumptions, claims):
    """For each consensus assumption: max NLI contradiction and entailment across the human's claims."""
    rows = []
    if not assumptions or not claims:
        return 0.0, rows
    for a in assumptions:
        probs = models.nli([(a["text"], c) for c in claims])
        j = int(probs[:, 0].argmax())
        rows.append(dict(assumption=a["text"], support=a["support"],
                         contradiction=round(float(probs[:, 0].max()), 3),
                         entailment=round(float(probs[:, 1].max()), 3),
                         strongest_challenge=claims[j]))
    w = np.array([r["support"] for r in rows], float)
    score = float(np.average([r["contradiction"] for r in rows], weights=w))
    return score, rows


def reasoning_quality(question, text, sents, q_vec, ans_vec):
    relevance = float(np.clip((float(q_vec @ ans_vec) - 0.15) / 0.40, 0, 1))
    ppl = models.perplexity(text)
    # Natural expository English is ~10-60 ppl under a 0.5B model; gibberish is >> 200.
    fluency = float(np.clip(1 - (math.log(ppl) - math.log(60)) / (math.log(400) - math.log(60)), 0, 1))
    if len(sents) >= 2:
        pairs = [(sents[i], sents[k]) for i in range(len(sents)) for k in range(i + 1, len(sents))][:20]
        contra = float(models.nli(pairs)[:, 0].max())
    else:
        contra = 0.0
    consistency = 1 - contra
    n_causal = len(re.findall(CAUSAL, text, flags=re.I))
    structure = _sat(n_causal, 2.0)
    words = len(text.split())
    depth = float(np.clip(words / 80, 0, 1)) * (1.0 if words <= 600 else 0.85)
    score = 0.30 * relevance + 0.20 * fluency + 0.20 * consistency + 0.20 * structure + 0.10 * depth
    # Fluency and relevance are necessary conditions, not just additive terms.
    score *= min(1.0, 0.25 + 0.75 * min(relevance, fluency) / 0.6)
    return float(np.clip(score, 0, 1)), dict(relevance=round(relevance, 3), fluency=round(fluency, 3),
                                              perplexity=round(ppl, 1), self_consistency=round(consistency, 3),
                                              causal_markers=n_causal, structure=round(structure, 3),
                                              words=words, depth=round(depth, 3))


def evidence_quality(text):
    hits = {k: len(re.findall(p, text, flags=re.I if k != "named examples" else 0)) for k, p in EVIDENCE_PATTERNS.items()}
    kinds = sum(1 for v in hits.values() if v > 0)
    score = 0.6 * (kinds / len(hits)) + 0.4 * _sat(sum(hits.values()), 4.0)
    return float(np.clip(score, 0, 1)), hits


def information_gain(ans_vec, pool_vecs):
    """Residual variance after projecting onto span(pool) = 1 - k^T K^-1 k
    (= exp of the log-det volume gain for unit vectors)."""
    if len(pool_vecs) == 0:
        return 1.0
    K = pool_vecs @ pool_vecs.T + 1e-3 * np.eye(len(pool_vecs))
    k = pool_vecs @ ans_vec
    resid = float(1 - k @ np.linalg.solve(K, k))
    return float(np.clip(resid / 0.6, 0, 1))  # 0.6 ~ residual of a fully off-topic answer (calibration)


# ---------------------------------------------------------------- main
def score_submission(question: str, answer: str, reasoning: str, baselines: list,
                     prior_texts: list | None = None) -> dict:
    t0 = time.time()
    prior_texts = prior_texts or []
    text = (answer.strip() + ("\n" + reasoning.strip() if reasoning.strip() else "")).strip()
    sents = split_sentences(text) or [text]

    q_vec, ans_vec = models.embed([question, text])
    base_vecs = models.embed(baselines)
    pool = baselines + prior_texts
    pool_vecs = models.embed(pool)

    div, div_meta = semantic_divergence(ans_vec, base_vecs)
    existing_sents = [s for b in pool for s in split_sentences(b)]
    cnov, claims = claim_novelty(sents, existing_sents)
    assumptions = consensus_assumptions(baselines)
    chal, chal_rows = assumption_challenge(assumptions, sents)
    novelty = 0.6 * cnov + 0.4 * chal
    rq, rq_meta = reasoning_quality(question, text, sents, q_vec, ans_vec)
    ev, ev_hits = evidence_quality(text)
    ig = information_gain(ans_vec, pool_vecs)

    gate = float(np.clip((rq - 0.25) / 0.35, 0, 1))
    raw = dict(semantic_divergence=div, conceptual_novelty=novelty, reasoning_quality=rq,
               evidence_quality=ev, information_gain=ig)
    credited = {k: v * (gate if k in GATED else 1.0) for k, v in raw.items()}
    contrib = {k: 100 * WEIGHTS[k] * credited[k] for k in WEIGHTS}
    total = float(sum(contrib.values()))
    originality = 100 * (0.30 * div + 0.25 * novelty + 0.10 * ig) / 0.65
    quality = 100 * (0.25 * rq + 0.10 * ev) / 0.35

    result = dict(
        score=round(total, 1), originality=round(originality, 1), quality=round(quality, 1),
        gate=round(gate, 3), components=raw, credited=credited, contributions=contrib,
        details=dict(semantic=div_meta, claim_novelty=round(cnov, 3), assumption_challenge=round(chal, 3),
                     reasoning=rq_meta, evidence=ev_hits, information_gain=round(ig, 3),
                     pool_size=len(pool)),
        claims=claims, assumptions=chal_rows,
        meta=dict(version=VERSION, weights=WEIGHTS, embedding=models.embedding_name(),
                  nli=models.NLI_NAME, lm=models.LLM_NAME, seconds=round(time.time() - t0, 2),
                  scored_at=time.strftime("%Y-%m-%dT%H:%M:%S")),
    )
    result["explanation"] = explain(result)
    return result


def explain(r: dict) -> list:
    c, d = r["components"], r["details"]
    out = []
    sim = d["semantic"]["max_baseline_similarity"]
    out.append(f"Semantic divergence {c['semantic_divergence']:.2f}: closest AI baseline has cosine similarity "
               f"{sim:.2f} ({'near-paraphrase' if sim > 0.75 else 'related but distinct' if sim > 0.45 else 'very different meaning'}).")
    new_claims = [x for x in r["claims"] if x["nearest_similarity"] < CLAIM_NEW_SIM]
    out.append(f"Conceptual novelty {c['conceptual_novelty']:.2f}: {len(new_claims)}/{len(r['claims'])} claims have no close "
               f"counterpart in existing answers; consensus-assumption challenge {d['assumption_challenge']:.2f}.")
    rq = d["reasoning"]
    out.append(f"Reasoning quality {c['reasoning_quality']:.2f}: relevance {rq['relevance']:.2f}, fluency {rq['fluency']:.2f} "
               f"(ppl {rq['perplexity']}), self-consistency {rq['self_consistency']:.2f}, {rq['causal_markers']} causal links.")
    found = [k for k, v in d["evidence"].items() if v]
    out.append(f"Evidence quality {c['evidence_quality']:.2f}: " + (f"found {', '.join(found)}." if found else "no quantities, examples, mechanisms or sources detected."))
    out.append(f"Information gain {c['information_gain']:.2f}: share of this answer's meaning not explained by the "
               f"{d['pool_size']} existing answers.")
    if r["gate"] < 0.999:
        out.append(f"Quality gate {r['gate']:.2f}: originality was credited at {r['gate']:.0%} because reasoning quality "
                   f"is below the threshold. Divergence without coherence is not rewarded.")
    return out
