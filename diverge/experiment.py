"""
DIVERGE compute-efficiency experiment (reproducible, small-scale PROXY).

Question: at equal model, optimizer and epochs, does training on a set of
*divergent* (diverse, non-redundant) reasoning examples reach better held-out
performance per unit of compute than training on *conventional*
(consensus-typical) examples?

Why a proxy: DIVERGE does not yet have enough human submissions to train on.
We instead use a public, human-written, gold-labeled commonsense reasoning
dataset (CommonsenseQA, Talmor et al. 2019) and apply the SAME embedding-based
divergence notion used by the DIVERGE Score to select training subsets.

Arms (identical model, identical hyper-parameters, identical budget N):
  conventional : N examples closest to the pool centroid (most typical)
  divergent    : N examples by greedy k-center / farthest-point selection
  random       : N uniformly random examples (control)

Model: frozen all-MiniLM-L6-v2 embeddings + identical 2-layer MLP choice scorer
(trained from scratch per run). Eval: accuracy on the official validation split
(1,221 held-out questions; chance = 20%).

Compute: wall-clock and process CPU seconds are MEASURED; training FLOPs are
computed analytically (6 x params x scored pairs). Energy = measured seconds x an
ASSUMED device power (configurable), reported separately as an estimate.

Run:  .venv/bin/python -m diverge.experiment
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "experiment"
EMB_CACHE = OUT / "csqa_embeddings.npz"
RESULTS = OUT / "results.json"

CONFIG = dict(
    dataset="tau/commonsense_qa", embedder="sentence-transformers/all-MiniLM-L6-v2",
    budgets=[100, 200, 400, 800, 1600, 3200], seeds=[0, 1, 2, 3, 4],
    arms=["conventional", "divergent", "random"],
    hidden=256, epochs=30, batch_size=32, lr=1e-3, weight_decay=1e-4,
    target_accuracies=[0.30, 0.33, 0.36],
)


# ---------------------------------------------------------------- data
def load_csqa():
    from huggingface_hub import hf_hub_download
    tr = pd.read_parquet(hf_hub_download(CONFIG["dataset"], "data/train-00000-of-00001.parquet", repo_type="dataset"))
    va = pd.read_parquet(hf_hub_download(CONFIG["dataset"], "data/validation-00000-of-00001.parquet", repo_type="dataset"))
    return tr, va


def _split(df):
    qs = df["question"].tolist()
    choices = [list(c["text"]) for c in df["choices"]]
    labels = np.array([list(c["label"]).index(k) for c, k in zip(df["choices"], df["answerKey"])])
    return qs, choices, labels


def embed_all():
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE)
        return {k: z[k] for k in z.files}
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(CONFIG["embedder"])
    tr, va = load_csqa()
    out = {}
    t = time.time()
    for name, df in [("train", tr), ("val", va)]:
        qs, ch, y = _split(df)
        out[f"{name}_q"] = m.encode(qs, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        flat = [c for cs in ch for c in cs]
        out[f"{name}_c"] = m.encode(flat, batch_size=256, normalize_embeddings=True,
                                    show_progress_bar=False).reshape(len(qs), 5, -1)
        out[f"{name}_y"] = y
    out["embed_seconds"] = np.array(time.time() - t)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB_CACHE, **out)
    return out


# ---------------------------------------------------------------- selection
def select(arm: str, Q: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if arm == "random":
        return rng.choice(len(Q), n, replace=False)
    if arm == "conventional":
        centroid = Q.mean(0)
        centroid /= np.linalg.norm(centroid)
        sims = Q @ centroid
        # seed-dependent tie-breaking jitter keeps runs comparable to the other arms
        return np.argsort(-(sims + rng.normal(0, 1e-6, len(sims))))[:n]
    if arm == "divergent":
        first = int(rng.integers(len(Q)))
        chosen = [first]
        mind = 1 - Q @ Q[first]
        for _ in range(n - 1):
            j = int(mind.argmax())
            chosen.append(j)
            mind = np.minimum(mind, 1 - Q @ Q[j])
        return np.array(chosen)
    raise ValueError(arm)


# ---------------------------------------------------------------- model
def _features(q, c):
    import torch
    qe = q[:, None, :].expand_as(c)
    return torch.cat([qe, c, qe * c, (qe - c).abs()], dim=-1)


def train_eval(Xq, Xc, y, Vq, Vc, vy, seed: int):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    d = Xq.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(4 * d, CONFIG["hidden"]), torch.nn.ReLU(),
                              torch.nn.Linear(CONFIG["hidden"], 1))
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    Xq, Xc, y = torch.tensor(Xq), torch.tensor(Xc), torch.tensor(y)
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    t0, c0 = time.perf_counter(), time.process_time()
    pairs = 0
    for _ in range(CONFIG["epochs"]):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, CONFIG["batch_size"]):
            idx = perm[i:i + CONFIG["batch_size"]]
            logits = net(_features(Xq[idx], Xc[idx])).squeeze(-1)
            loss = torch.nn.functional.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            pairs += len(idx) * 5
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    with torch.no_grad():
        pred = net(_features(torch.tensor(Vq), torch.tensor(Vc))).squeeze(-1).argmax(-1).numpy()
    return dict(accuracy=float((pred == vy).mean()), train_seconds=wall, train_cpu_seconds=cpu,
                train_flops=float(6 * params * pairs), params=int(params))


def redundancy(Q_sel: np.ndarray) -> float:
    """Mean pairwise cosine similarity of the selected questions (lower = more divergent)."""
    S = Q_sel @ Q_sel.T
    n = len(S)
    return float((S.sum() - n) / (n * (n - 1)))


# ---------------------------------------------------------------- run
def run(config_overrides: dict | None = None, progress=None) -> dict:
    cfg = dict(CONFIG, **(config_overrides or {}))
    CONFIG.update(cfg)
    E = embed_all()
    Q, C, Y = E["train_q"], E["train_c"], E["train_y"]
    Vq, Vc, vy = E["val_q"], E["val_c"], E["val_y"]
    rows = []
    total = len(cfg["arms"]) * len(cfg["budgets"]) * len(cfg["seeds"])
    k = 0
    for n in cfg["budgets"]:
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                t = time.perf_counter()
                idx = select(arm, Q, n, seed)
                sel_s = time.perf_counter() - t
                r = train_eval(Q[idx], C[idx], Y[idx], Vq, Vc, vy, seed)
                r.update(arm=arm, budget=n, seed=seed, selection_seconds=sel_s,
                         redundancy=redundancy(Q[idx]))
                rows.append(r)
                k += 1
                if progress:
                    progress(k / total, f"{arm} N={n} seed={seed}: acc {r['accuracy']:.3f}")
    res = dict(
        config=cfg, rows=rows,
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         processor=platform.processor(), torch=__import__("torch").__version__,
                         device="cpu"),
        embedding_seconds_one_time=float(E["embed_seconds"]),
        n_train_pool=int(len(Q)), n_heldout=int(len(vy)),
        completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2))
    return res


def load_results() -> dict | None:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else None


def summarize(res: dict) -> pd.DataFrame:
    df = pd.DataFrame(res["rows"])
    g = df.groupby(["arm", "budget"])
    s = g.agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
              train_s=("train_seconds", "mean"), cpu_s=("train_cpu_seconds", "mean"),
              flops=("train_flops", "mean"), selection_s=("selection_seconds", "mean"),
              redundancy=("redundancy", "mean"), runs=("accuracy", "count")).reset_index()
    s["acc_per_pflop"] = (s.acc_mean - 0.2) / (s.flops / 1e15)  # accuracy above chance per PFLOP
    return s


def compute_to_target(summary: pd.DataFrame, target: float) -> pd.DataFrame:
    """Smallest budget whose MEAN accuracy reaches `target` (no interpolation)."""
    out = []
    for arm, g in summary.groupby("arm"):
        hit = g[g.acc_mean >= target].sort_values("budget")
        if len(hit):
            h = hit.iloc[0]
            out.append(dict(arm=arm, target=target, budget=int(h.budget), flops=h.flops,
                            train_s=h.train_s, cpu_s=h.cpu_s))
        else:
            out.append(dict(arm=arm, target=target, budget=None, flops=np.nan, train_s=np.nan, cpu_s=np.nan))
    return pd.DataFrame(out)


def significance(res: dict, a: str = "divergent", b: str = "conventional") -> pd.DataFrame:
    """Per budget: mean accuracy difference and a paired sign-flip permutation p-value over seeds."""
    df = pd.DataFrame(res["rows"])
    out = []
    rng = np.random.default_rng(0)
    for n, g in df.groupby("budget"):
        x = g[g.arm == a].sort_values("seed").accuracy.to_numpy()
        z = g[g.arm == b].sort_values("seed").accuracy.to_numpy()
        if len(x) != len(z) or not len(x):
            continue
        d = x - z
        obs = abs(d.mean())
        flips = rng.choice([-1, 1], size=(10000, len(d)))
        p = float(((np.abs((flips * d).mean(1))) >= obs - 1e-12).mean())
        out.append(dict(budget=int(n), diff=float(d.mean()), p_value=p, seeds=len(d)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    t = time.time()
    r = run(progress=lambda f, m: print(f"[{f:5.0%}] {m}", flush=True))
    s = summarize(r)
    print(s.round(4).to_string())
    print(significance(r).round(4).to_string())
    print(f"done in {time.time() - t:.0f}s -> {RESULTS}")
