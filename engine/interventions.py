"""
Intervention catalogue and MW-impact evaluation.

Effect sizes and feasibility scores are SIMULATED priors for demonstration.
They are not estimates from engineering studies.

Each intervention is a list of commutative parameter operations, so any
subset can be applied in any order. Impact is measured two ways:
  * standalone  : MW gained when applied alone to the current scenario
  * Shapley     : average marginal MW across random orderings of all
                  interventions (fair credit when bottlenecks interact)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import solve


@dataclass(frozen=True)
class Intervention:
    id: str
    name: str
    category: str
    description: str
    ops: tuple
    feasibility: float  # simulated prior, 0-1
    consensus: bool     # is this a mainstream / widely discussed lever?

    def apply(self, p: dict) -> dict:
        for op in self.ops:
            kind = op[0]
            if kind == "add":
                p[op[1]] += op[2]
            elif kind == "mul":
                p[op[1]] *= op[2]
            elif kind == "max":
                p[op[1]] = max(p[op[1]], op[2])
            elif kind == "zone_add":
                d = dict(p.get(op[1], {}))
                d[op[2]] = d.get(op[2], 0.0) + op[3]
                p[op[1]] = d
            elif kind == "add_if_year":  # only counts if online by the target year
                if p["target_year"] >= op[3]:
                    p[op[1]] += op[2]
        return p


INTERVENTIONS = [
    Intervention("gas", "New gas peaking plant", "Peak demand",
                 "Build new natural gas power plant generation capacity to supply reliable power for growing data center electricity demand.",
                 (("add", "peak_margin_mw", 300.0),), 0.50, True),
    Intervention("osw", "Offshore wind PPAs", "Peak demand",
                 "Sign renewable energy power purchase agreements for new offshore wind generation to supply clean electricity to data centers.",
                 (("add", "peak_margin_mw", 150.0),), 0.45, True),
    Intervention("smr", "Small modular reactors", "Peak demand",
                 "Build nuclear small modular reactors to provide clean firm generation and power supply for data centers.",
                 (("add_if_year", "peak_margin_mw", 300.0, 2033),), 0.15, True),
    Intervention("permit", "Permitting reform", "Interconnection throughput",
                 "Permitting reform to speed construction approval of new generation and transmission lines.",
                 (("add", "ic_bonus_mw", 60.0), ("mul", "grid_multiplier", 1.05)), 0.40, True),
    Intervention("ic_auto", "GPU-accelerated interconnection studies", "Interconnection throughput",
                 "Replace serial manual power-flow queue studies with batched GPU contingency simulation so each engineer clears twice the queue megawatts per year.",
                 (("mul", "ic_mw_per_engineer_year", 2.0),), 0.70, False),
    Intervention("ic_hire", "Expand interconnection engineering staff", "Interconnection throughput",
                 "Hire and train additional utility interconnection study engineers to process the queue.",
                 (("add", "ic_engineers", 6),), 0.60, False),
    Intervention("dlr", "Dynamic line ratings", "Grid capacity",
                 "Weather-aware sensors rate conductors by real-time ambient cooling instead of static worst-case assumptions, revealing latent headroom on existing wires.",
                 (("add", "dlr_uplift", 0.15),), 0.75, False),
    Intervention("recond", "Advanced-conductor reconductoring", "Grid capacity",
                 "Restring existing corridors with composite-core high-temperature low-sag conductors, raising thermal limits without new rights-of-way.",
                 (("mul", "grid_multiplier", 1.25),), 0.55, False),
    Intervention("flex", "Peak-aware AI training scheduler", "Flexible workloads",
                 "Pause or migrate checkpointed training jobs during the few annual system peak hours under contractually committed curtailable load agreements.",
                 (("max", "flex_share", 0.50), ("max", "peak_curtailment", 0.90)), 0.65, False),
    Intervention("btm", "On-site batteries at data centers", "Peak demand",
                 "Behind-the-meter battery storage discharges at peak so the facility draws less from constrained feeders and the bulk system.",
                 (("add", "btm_storage_mw", 150.0),), 0.60, False),
    Intervention("xf_refurb", "Transformer refurbishment & spare pooling", "Transformer availability",
                 "Rewind and requalify retired utility transformers and pool spares across utilities to shortcut multi-year manufacturing lead times.",
                 (("mul", "xfmr_existing_multiplier", 1.20),), 0.55, False),
    Intervention("xf_bulk", "Standardized bulk transformer procurement", "Transformer availability",
                 "Aggregate orders under standardized specifications so manufacturers can commit production slots earlier.",
                 (("add", "xfmr_units_per_year", 1.5),), 0.50, False),
    Intervention("brownfield", "Reuse retired plant interconnection points", "Interconnection throughput",
                 "Site campuses at retired fossil plant switchyards, reusing existing high-voltage interconnection rights through surplus interconnection service instead of joining the queue.",
                 (("zone_add", "zone_grid_bonus", "Southeastern MA", 200.0),
                  ("zone_add", "zone_xfmr_bonus", "Southeastern MA", 150.0),
                  ("add", "ic_bonus_mw", 150.0)), 0.45, False),
    Intervention("heat", "Waste heat to district heating", "Peak demand",
                 "Export server waste heat into thermal networks, displacing electric heat pump load on the winter system peak.",
                 (("add", "peak_margin_mw", 25.0),), 0.50, False),
]
BY_ID = {iv.id: iv for iv in INTERVENTIONS}


def describe_ops(iv: Intervention) -> str:
    parts = []
    for op in iv.ops:
        if op[0] == "add":
            parts.append(f"{op[1]} +{op[2]:g}")
        elif op[0] == "mul":
            parts.append(f"{op[1]} ×{op[2]:g}")
        elif op[0] == "max":
            parts.append(f"{op[1]} ≥ {op[2]:g}")
        elif op[0] == "zone_add":
            parts.append(f"{op[1]}[{op[2]}] +{op[3]:g}")
        elif op[0] == "add_if_year":
            parts.append(f"{op[1]} +{op[2]:g} (if target ≥ {op[3]})")
    return "; ".join(parts)


def apply_set(params: dict, ids) -> dict:
    p = copy.deepcopy(params)
    for iid in ids:
        BY_ID[iid].apply(p)
    return p


def evaluate(params: dict, n_perm: int = 40, seed: int = 7) -> pd.DataFrame:
    memo: dict = {}

    def value(ids: frozenset) -> float:
        if ids not in memo:
            memo[ids] = solve(apply_set(params, sorted(ids))).total_mw
        return memo[ids]

    base = value(frozenset())
    ids = [iv.id for iv in INTERVENTIONS]
    standalone = {i: value(frozenset([i])) - base for i in ids}

    rng = np.random.default_rng(seed)
    shap = {i: 0.0 for i in ids}
    for _ in range(n_perm):
        cur: set = set()
        prev = base
        for i in rng.permutation(ids):
            cur.add(i)
            v = value(frozenset(cur))
            shap[i] += v - prev
            prev = v
    shap = {i: s / n_perm for i, s in shap.items()}

    rows = []
    for iv in INTERVENTIONS:
        rows.append(dict(
            id=iv.id, intervention=iv.name, category=iv.category,
            standalone_mw=standalone[iv.id], shapley_mw=shap[iv.id],
            feasibility=iv.feasibility, consensus=iv.consensus,
            effect=describe_ops(iv), description=iv.description,
        ))
    df = pd.DataFrame(rows)
    df.attrs["base_mw"] = base
    df.attrs["all_mw"] = value(frozenset(ids))
    return df
