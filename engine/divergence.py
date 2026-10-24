"""
Divergence Score: ranks interventions that are (a) semantically far from
conventional energy interventions, (b) high simulated MW impact, and
(c) technically feasible.

Semantic novelty
----------------
Each text is embedded with a sentence-transformers model
(all-MiniLM-L6-v2, normalized). Novelty is the cosine distance to the
conventional baseline:

    novelty = 1 - mean(top-k cosine similarity to baseline statements)

This is an absolute scale (not min-max scaled), so a newly submitted
hypothesis can be compared with the catalogue. If sentence-transformers is
unavailable the module falls back to TF-IDF and reports that via
`embedding_method()`.

Divergence = 100 * weighted geometric mean(novelty, impact, feasibility)
A geometric mean means an idea must score on all three axes.
"""
from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# Baseline of conventional energy interventions: the mainstream answers to
# "how do we power data-center growth". Hand-written by the UNBOUND team as a
# reference set; it is not drawn from a survey.
CONVENTIONAL_BASELINE = [
    "Build more electricity generation to power new data centers.",
    "Build new natural gas combined-cycle and peaking power plants.",
    "Sign renewable energy power purchase agreements for solar and wind.",
    "Procure offshore wind power to supply clean electricity.",
    "Build nuclear power plants and small modular reactors for clean firm power.",
    "Extend the life of existing nuclear plants.",
    "Permitting reform to speed construction of energy infrastructure.",
    "Build new high-voltage transmission lines to expand grid capacity.",
    "Upgrade substations and install new transformers.",
    "Utilities invest in grid modernization and distribution upgrades.",
    "Improve data center energy efficiency and cooling to reduce power usage.",
    "Buy renewable energy certificates and carbon offsets.",
    "Install rooftop solar panels on data center buildings.",
    "Increase capacity market procurement to ensure resource adequacy.",
    "Import more hydropower from Canada.",
    "Deploy grid-scale battery storage to balance renewables.",
    "Use backup diesel generators for data center reliability.",
    "Raise electricity rates for large loads to pay for grid upgrades.",
    "Hire more utility workers and train electricians.",
    "Locate data centers where land and power are cheap.",
]

_STOP = set("""a an and the to of for in on by with from into at so as is are be this that
than instead each per under through their its it or not new""".split())


# ---------------------------------------------------------------- embeddings
@functools.lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)
    except Exception:  # missing package / offline without cached weights
        return None


def embedding_method() -> str:
    return f"sentence-transformers · {MODEL_NAME.split('/')[-1]}" if _model() is not None \
        else "TF-IDF fallback (sentence-transformers unavailable)"


@functools.lru_cache(maxsize=4096)
def _embed_one(text: str) -> tuple:
    return tuple(_model().encode([text], normalize_embeddings=True)[0].tolist())


def _tokens(text: str):
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOP and len(t) > 2]


def _tfidf(docs):
    toks = [_tokens(d) for d in docs]
    vocab = sorted({t for ts in toks for t in ts})
    idx = {t: i for i, t in enumerate(vocab)}
    tf = np.zeros((len(docs), max(len(vocab), 1)))
    for r, ts in enumerate(toks):
        for t in ts:
            tf[r, idx[t]] += 1
    df = (tf > 0).sum(0)
    m = tf * (np.log((1 + len(docs)) / (1 + df)) + 1)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def _similarity(texts) -> np.ndarray:
    """Cosine similarity matrix: texts x baseline."""
    texts = list(texts)
    if _model() is not None:
        x = np.array([_embed_one(t) for t in texts])
        b = np.array([_embed_one(t) for t in CONVENTIONAL_BASELINE])
        return x @ b.T
    m = _tfidf(CONVENTIONAL_BASELINE + texts)
    return m[len(CONVENTIONAL_BASELINE):] @ m[: len(CONVENTIONAL_BASELINE)].T


def semantic_novelty(texts) -> pd.DataFrame:
    """Novelty in [0, 1] plus the nearest conventional statement for each text."""
    sim = _similarity(texts)
    k = min(TOP_K, sim.shape[1])
    topk = np.sort(sim, axis=1)[:, -k:]
    nov = np.clip(1 - topk.mean(axis=1), 0, 1)
    nearest = sim.argmax(axis=1)
    return pd.DataFrame({
        "novelty": nov,
        "nearest_conventional": [CONVENTIONAL_BASELINE[j] for j in nearest],
        "nearest_similarity": sim.max(axis=1),
    })


def novelty_scores(descriptions) -> np.ndarray:
    return semantic_novelty(descriptions)["novelty"].to_numpy()


# ---------------------------------------------------------------- scoring
def geometric_divergence(novelty, impact, feasibility, w_nov=1.0, w_imp=1.0, w_feas=1.0):
    w = np.array([w_nov, w_imp, w_feas], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones(3) / 3
    eps = 1e-3
    logs = (w[0] * np.log(np.asarray(novelty, float) + eps)
            + w[1] * np.log(np.asarray(impact, float) + eps)
            + w[2] * np.log(np.asarray(feasibility, float) + eps))
    return 100 * np.exp(logs)


def divergence_scores(df: pd.DataFrame, impact_col: str, w_nov: float, w_imp: float,
                      w_feas: float) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    sem = semantic_novelty(out["description"])
    out["novelty"] = sem["novelty"]
    out["nearest_conventional"] = sem["nearest_conventional"]
    imp = out[impact_col].clip(lower=0)
    out["impact"] = imp / imp.max() if imp.max() > 0 else 0.0
    out["divergence"] = geometric_divergence(out["novelty"], out["impact"], out["feasibility"],
                                             w_nov, w_imp, w_feas)
    return out.sort_values("divergence", ascending=False).reset_index(drop=True)
