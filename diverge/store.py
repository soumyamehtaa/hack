"""Submission store (JSONL), leaderboard, and training-data export."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS = ROOT / "data" / "submissions.jsonl"
_lock = threading.Lock()

TRAINING_THRESHOLD = 45.0  # default minimum score to be "training-grade"; adjustable in the UI


def load_all() -> list:
    if not SUBMISSIONS.exists():
        return []
    out = []
    for line in SUBMISSIONS.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def add(record: dict) -> dict:
    record = dict(record)
    record.setdefault("id", uuid.uuid4().hex[:10])
    record.setdefault("submitted_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    SUBMISSIONS.parent.mkdir(parents=True, exist_ok=True)
    with _lock, SUBMISSIONS.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def prior_texts(question_id: str) -> list:
    return [f"{r['answer']}\n{r.get('reasoning', '')}".strip()
            for r in load_all() if r["question_id"] == question_id]


def table(records: list | None = None) -> pd.DataFrame:
    recs = load_all() if records is None else records
    if not recs:
        return pd.DataFrame()
    rows = []
    for r in recs:
        s = r["evaluation"]
        rows.append(dict(id=r["id"], user=r["user"], question_id=r["question_id"], domain=r.get("domain", ""),
                         question=r["question"], answer=r["answer"], score=s["score"],
                         originality=s["originality"], quality=s["quality"], gate=s["gate"],
                         demo_seed=r.get("demo_seed", False), submitted_at=r["submitted_at"],
                         **{k: v for k, v in s["components"].items()}))
    return pd.DataFrame(rows)


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby("user").agg(best=("score", "max"), mean=("score", "mean"), submissions=("score", "count"),
                               originality=("originality", "mean"), quality=("quality", "mean"),
                               training_grade=("score", lambda s: int((s >= TRAINING_THRESHOLD).sum())),
                               demo=("demo_seed", "max"))
    return g.sort_values(["best", "mean"], ascending=False).reset_index()


# ---------------------------------------------------------------- export
def to_training_record(r: dict, fmt: str) -> dict:
    ev = r["evaluation"]
    meta = dict(id=r["id"], user=r["user"], submitted_at=r["submitted_at"], question_id=r["question_id"],
                demo_seed=r.get("demo_seed", False), scorer=ev["meta"], gate=ev["gate"],
                baseline_generation=r.get("baseline_meta", {}))
    if fmt == "chat (fine-tuning)":
        return dict(messages=[
            dict(role="user", content=r["question"]),
            dict(role="assistant", content=(r["answer"] + ("\n\nReasoning: " + r["reasoning"] if r.get("reasoning") else "")))],
            metadata=dict(divergence_score=ev["score"], components=ev["components"], **meta))
    if fmt == "preference pair (DPO)":
        return dict(prompt=r["question"],
                    chosen=r["answer"] + ("\n\nReasoning: " + r["reasoning"] if r.get("reasoning") else ""),
                    rejected=r["baselines"][0] if r.get("baselines") else "",
                    metadata=dict(divergence_score=ev["score"], components=ev["components"], **meta))
    return dict(question=r["question"], human_response=r["answer"], reasoning=r.get("reasoning", ""),
                ai_baselines=r.get("baselines", []), divergence_score=ev["score"],
                originality=ev["originality"], quality=ev["quality"],
                components=ev["components"], credited_components=ev["credited"],
                consensus_assumptions=ev.get("assumptions", []), explanation=ev["explanation"],
                evaluation_metadata=meta)


def export_jsonl(ids: list, fmt: str) -> str:
    by_id = {r["id"]: r for r in load_all()}
    return "\n".join(json.dumps(to_training_record(by_id[i], fmt), ensure_ascii=False)
                     for i in ids if i in by_id) + "\n"
