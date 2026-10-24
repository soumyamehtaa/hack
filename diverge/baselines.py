"""
AI baseline answers ("conventional reasoning") and consensus-assumption extraction.

Baselines are sampled from a local LLM with a prompt that asks for the most
widely accepted explanation. They are cached in data/baselines.json with
generation metadata so every score is reproducible.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from . import models

CACHE = Path(__file__).resolve().parent.parent / "data" / "baselines.json"
N_BASELINES = 3
PROMPT = ("Answer the following question in 3 to 4 sentences, giving the most widely accepted, "
          "conventional explanation.\n\nQuestion: {q}")
GEN_KW = dict(max_new_tokens=140, temperature=0.8, seed=7)


def _load() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _save(d: dict):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, indent=2))


def get_baselines(question: str, generate_if_missing: bool = True) -> dict | None:
    d = _load()
    if question in d:
        return d[question]
    if not generate_if_missing:
        return None
    t = time.time()
    texts = models.generate(PROMPT.format(q=question), n=N_BASELINES, **GEN_KW)
    rec = dict(question=question, baselines=texts,
               meta=dict(model=models.LLM_NAME, prompt=PROMPT, n=N_BASELINES, **GEN_KW,
                         generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                         seconds=round(time.time() - t, 2)))
    d = _load()
    d[question] = rec
    _save(d)
    return rec


# ---------------------------------------------------------------- sentences & assumptions
def split_sentences(text: str) -> list:
    text = re.sub(r"\*\*|__|#+", "", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for s in parts:
        s = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", s).strip()
        if len(s.split()) >= 4:
            out.append(s)
    return out


def consensus_assumptions(baselines: list, sim_threshold: float = 0.6, max_k: int = 5) -> list:
    """Claims that recur across >= 2 independent baseline samples.

    Returns list of dicts {text, support, members}: support = number of
    baselines containing a semantically equivalent sentence (cos >= threshold).
    """
    sents, owner = [], []
    for j, b in enumerate(baselines):
        for s in split_sentences(b):
            sents.append(s)
            owner.append(j)
    if not sents:
        return []
    E = models.embed(sents)
    S = E @ E.T
    owner = np.array(owner)
    support = []
    for i in range(len(sents)):
        others = {owner[k] for k in np.where(S[i] >= sim_threshold)[0] if owner[k] != owner[i]}
        support.append(1 + len(others))
    support = np.array(support)
    centrality = np.array([S[i][owner != owner[i]].mean() if (owner != owner[i]).any() else 0
                           for i in range(len(sents))])
    order = np.lexsort((-centrality, -support))
    chosen = []
    for i in order:
        if len(chosen) >= max_k:
            break
        if any(S[i, c] >= sim_threshold for c in chosen):
            continue
        if support[i] < 2 and len(chosen) >= 2:
            continue  # after two, only keep genuinely shared claims
        chosen.append(i)
    return [dict(text=sents[i], support=int(support[i]), centrality=float(centrality[i]))
            for i in chosen]
