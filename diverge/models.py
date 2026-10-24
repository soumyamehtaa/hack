"""
Lazy-loaded local models (no API keys, no data leaves the machine):
  * Embeddings : sentence-transformers all-MiniLM-L6-v2 (reused from engine.divergence)
  * NLI        : cross-encoder/nli-deberta-v3-xsmall (contradiction / entailment)
  * LLM        : Qwen/Qwen2.5-0.5B-Instruct (baseline answers + fluency/perplexity)
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NLI_NAME = "cross-encoder/nli-deberta-v3-xsmall"


# ---------------------------------------------------------------- embeddings (reuse UNBOUND engine)
def embed(texts) -> np.ndarray:
    from engine.divergence import _embed_one, _model
    if _model() is None:
        raise RuntimeError("sentence-transformers model unavailable")
    return np.array([_embed_one(t) for t in texts], dtype=np.float32)


def embedding_name() -> str:
    from engine.divergence import embedding_method
    return embedding_method()


# ---------------------------------------------------------------- NLI
@functools.lru_cache(maxsize=1)
def _nli():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(NLI_NAME)


@functools.lru_cache(maxsize=8192)
def _nli_pair(premise: str, hypothesis: str) -> tuple:
    p = _nli().predict([(premise, hypothesis)], apply_softmax=True)[0]
    return tuple(float(x) for x in p)  # (contradiction, entailment, neutral)


def nli(pairs) -> np.ndarray:
    """Returns array [n, 3] of (contradiction, entailment, neutral) probabilities."""
    if not pairs:
        return np.zeros((0, 3))
    return np.array([_nli_pair(a[:600], b[:600]) for a, b in pairs])


# ---------------------------------------------------------------- LLM
def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


@functools.lru_cache(maxsize=1)
def _llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    dev = _device()
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, dtype=torch.float16 if dev == "mps" else torch.float32).to(dev)
    model.eval()
    return tok, model, dev


def generate(prompt: str, n: int = 3, max_new_tokens: int = 140, temperature: float = 0.8,
             seed: int = 0) -> list:
    import torch
    tok, model, dev = _llm()
    msgs = [{"role": "system", "content": "You are a helpful, knowledgeable assistant."},
            {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(dev)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=0.9, num_return_sequences=n,
                             pad_token_id=tok.eos_token_id)
    L = enc["input_ids"].shape[1]
    return [tok.decode(o[L:], skip_special_tokens=True).strip() for o in out]


@functools.lru_cache(maxsize=2048)
def perplexity(text: str) -> float:
    """Per-token perplexity of `text` under the local LLM (fluency signal)."""
    import torch
    tok, model, dev = _llm()
    ids = tok(text[:2000], return_tensors="pt").input_ids.to(dev)
    if ids.shape[1] < 2:
        return 1e4
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.float().item()
    return float(math.exp(min(loss, 20)))
