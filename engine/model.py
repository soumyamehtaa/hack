"""
Linear-programming bottleneck engine.

Decision variables (per zone i):
    f_i  firm (inflexible) data-center load served, MW
    s_i  flexible (curtailable at peak) data-center load served, MW
    n_i  new transformer capacity allocated to zone i, MW

Objective: maximize  sum_i (f_i + s_i)   (deliverable data-center MW)

Constraints (A x <= b):
    Grid capacity        f_i + (1-c) s_i       <= G_i (1+DLR) * mult + bonus + storage share
    Transformers (zone)  f_i + s_i - n_i       <= E_i
    Transformers (pool)  sum n_i               <= units/yr * years * MVA * pf
    Interconnection      sum (f_i + s_i)       <= engineers * MW/eng-yr * years + bonus
    Peak demand          sum (f_i + (1-c) s_i) <= peak margin + storage
    Flexible share       (1-phi) s_i - phi f_i <= 0
    Demand pipeline      f_i + s_i             <= requested MW

Shadow prices (dual values from HiGHS) give d(MW served)/d(rhs): how many
additional MW of data-center load one more unit of each constraint unlocks.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

CATEGORIES = [
    "Grid capacity",
    "Transformer availability",
    "Interconnection throughput",
    "Peak demand",
    "Flexible workloads",
    "Demand pipeline",
]
SUPPLY_CATEGORIES = [c for c in CATEGORIES if c != "Demand pipeline"]

_TOL = 1e-4


@dataclass
class Solution:
    ok: bool
    total_mw: float
    zones: list
    firm: np.ndarray
    flex: np.ndarray
    new_xfmr: np.ndarray
    constraints: pd.DataFrame
    derived: dict = field(default_factory=dict)


def derived(params: dict) -> dict:
    years = max(params["target_year"] - params["base_year"], 0.25)
    ic = params["ic_engineers"] * params["ic_mw_per_engineer_year"] * years + params["ic_bonus_mw"]
    pool = (params["xfmr_units_per_year"] * years * params["xfmr_mva_per_unit"]
            * params["power_factor"])
    return dict(years=years, ic_capacity_mw=ic, xfmr_pool_mw=pool)


def solve(params: dict, removed: frozenset = frozenset()) -> Solution:
    zones = params["zones"]
    Z = len(zones)
    nv = 3 * Z
    d = derived(params)
    phi = float(np.clip(params["flex_share"], 0, 1))
    cur = float(np.clip(params["peak_curtailment"], 0, 1))
    storage = params["btm_storage_mw"]
    total_pipe = sum(z["pipeline_mw"] for z in zones) or 1.0
    gbonus = params.get("zone_grid_bonus", {})
    xbonus = params.get("zone_xfmr_bonus", {})

    rows, rhs, meta = [], [], []

    def add(coefs, b, name, cat, zone):
        if cat in removed:
            return
        r = np.zeros(nv)
        for j, v in coefs:
            r[j] += v
        rows.append(r)
        rhs.append(float(b))
        meta.append((name, cat, zone))

    for i, z in enumerate(zones):
        f, s, n = i, Z + i, 2 * Z + i
        share = z["pipeline_mw"] / total_pipe
        G = (z["grid_headroom_mw"] * params["grid_multiplier"] * (1 + params["dlr_uplift"])
             + gbonus.get(z["name"], 0.0) + storage * share)
        add([(f, 1), (s, 1 - cur)], G, f"Grid headroom · {z['name']}", "Grid capacity", z["name"])
        E = z["substation_headroom_mw"] * params["xfmr_existing_multiplier"] + xbonus.get(z["name"], 0.0)
        add([(f, 1), (s, 1), (n, -1)], E, f"Substation transformers · {z['name']}",
            "Transformer availability", z["name"])
        add([(f, -phi), (s, 1 - phi)], 0.0, f"Flexible-load share · {z['name']}",
            "Flexible workloads", z["name"])
        add([(f, 1), (s, 1)], z["pipeline_mw"], f"Demand pipeline · {z['name']}",
            "Demand pipeline", z["name"])

    add([(2 * Z + i, 1) for i in range(Z)], d["xfmr_pool_mw"],
        "New transformer deliveries (statewide)", "Transformer availability", "Statewide")
    add([(i, 1) for i in range(2 * Z)], d["ic_capacity_mw"],
        "Interconnection study throughput (statewide)", "Interconnection throughput", "Statewide")
    add([(i, 1) for i in range(Z)] + [(Z + i, 1 - cur) for i in range(Z)],
        params["peak_margin_mw"] + storage,
        "Peak capacity margin (statewide)", "Peak demand", "Statewide")

    c = np.zeros(nv)
    c[: 2 * Z] = -1.0
    c[2 * Z:] = 1e-7  # tiny cost so unused transformer allocation stays at zero

    A = np.vstack(rows)
    b = np.array(rhs)
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * nv, method="highs")
    names = [z["name"] for z in zones]

    if res.status != 0:
        empty = pd.DataFrame(columns=["constraint", "category", "zone", "rhs", "lhs",
                                      "slack", "shadow_price", "utilization", "binding"])
        return Solution(False, 0.0, names, np.zeros(Z), np.zeros(Z), np.zeros(Z), empty, d)

    x = res.x
    lhs = A @ x
    shadow = np.clip(-res.ineqlin.marginals, 0, None)
    slack = b - lhs
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(b > 1e-9, lhs / b, np.nan)
    df = pd.DataFrame(meta, columns=["constraint", "category", "zone"])
    df["rhs"], df["lhs"], df["slack"] = b, lhs, slack
    df["shadow_price"] = np.where(shadow > _TOL, shadow, 0.0)
    df["utilization"] = util
    df["binding"] = (np.abs(slack) <= 1e-6 * np.maximum(1, np.abs(b))) & (df["shadow_price"] > 0)

    return Solution(True, float(x[: 2 * Z].sum()), names, x[:Z], x[Z:2 * Z], x[2 * Z:], df, d)


def category_summary(sol: Solution) -> pd.DataFrame:
    df = sol.constraints
    out = []
    for cat in CATEGORIES:
        sub = df[df.category == cat]
        out.append(dict(
            category=cat,
            shadow_price=float(sub.shadow_price.sum()) if len(sub) else 0.0,
            max_utilization=float(np.nanmax(sub.utilization)) if sub.utilization.notna().any() else np.nan,
            binding=int(sub.binding.sum()),
        ))
    return pd.DataFrame(out)


def primary_binding(sol: Solution) -> str:
    """Supply-side category with the largest aggregate shadow price."""
    cs = category_summary(sol)
    sup = cs[cs.category.isin(SUPPLY_CATEGORIES)].sort_values("shadow_price", ascending=False)
    if len(sup) and sup.iloc[0].shadow_price > _TOL:
        return sup.iloc[0].category
    return "Demand pipeline (fully served)"


def cascade(params: dict, max_steps: int = 6) -> list:
    """Remove the binding constraint that unlocks the most MW, re-solve,
    and record which constraint becomes binding next."""
    removed: set = set()
    steps = []
    sol = solve(params)
    steps.append(dict(step=0, action="Baseline", total_mw=sol.total_mw,
                      exposed=None, removed=[]))
    for k in range(1, max_steps + 1):
        cs = category_summary(sol)
        candidates = cs[(cs.category.isin(SUPPLY_CATEGORIES)) & (cs.shadow_price > _TOL)
                        & (~cs.category.isin(removed))]
        if candidates.empty:
            break
        best, best_gain, best_sol = None, -1.0, None
        for cat in candidates.category:
            trial = solve(params, frozenset(removed | {cat}))
            gain = trial.total_mw - sol.total_mw
            if gain > best_gain + 1e-6:
                best, best_gain, best_sol = cat, gain, trial
        steps[-1]["exposed"] = best
        removed.add(best)
        sol = best_sol
        steps.append(dict(step=k, action=f"Remove {best}", total_mw=sol.total_mw,
                          exposed=None, removed=sorted(removed), gain=best_gain))
    steps[-1]["exposed"] = primary_binding(sol)
    return steps


def sweep(params: dict, key: str, values) -> pd.DataFrame:
    out = []
    for v in values:
        p = copy.deepcopy(params)
        p[key] = float(v)
        sol = solve(p)
        out.append(dict(value=float(v), total_mw=sol.total_mw, binding=primary_binding(sol)))
    return pd.DataFrame(out)
