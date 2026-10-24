"""
Consensus Breaker: three personas propose an intervention for the SAME
binding bottleneck; every proposal is scored by the deterministic LP engine.

Two proposal sources:
  * "scripted" (default): a hand-written DEMONSTRATION library, one proposal
    per persona per bottleneck category. Not AI-generated.
  * "llm": if ANTHROPIC_API_KEY or OPENAI_API_KEY is set and the user opts in,
    each persona is prompted for a JSON proposal. Only the bottleneck summary
    and parameter list are sent (no code, no user data).

In both modes, proposals are expressed as bounded operations on the model's
parameters, so MW unlocked is always computed by `engine.model.solve`, never
claimed by the persona. Effect magnitudes, feasibility and cost are
assumptions (SIMULATED for the scripted library).
"""
from __future__ import annotations

import copy
import json
import os
import re

import numpy as np
import pandas as pd
import requests

from .divergence import geometric_divergence, semantic_novelty
from .interventions import Intervention, describe_ops
from .model import primary_binding, solve

PERSONAS = {
    "Grid Engineer": dict(icon="⚡", color="#4EA8DE",
                          brief="Utility planning engineer. Trusts proven equipment and standards; minimizes reliability risk."),
    "Energy Economist": dict(icon="◈", color="#E9C46A",
                             brief="Market designer. Fixes bottlenecks with prices, tariffs, contracts and incentives."),
    "Contrarian Researcher": dict(icon="◐", color="#B388FF",
                                  brief="Challenges the framing of the problem; prefers unconventional, software- or "
                                        "architecture-level fixes."),
}

# Parameters a proposal may touch, with hard bounds on the resulting value.
PARAM_BOUNDS = {
    "ic_engineers": (1, 60), "ic_mw_per_engineer_year": (1, 120), "ic_bonus_mw": (0, 600),
    "xfmr_units_per_year": (0, 12), "xfmr_existing_multiplier": (0.3, 2.5),
    "grid_multiplier": (0.3, 2.5), "dlr_uplift": (0, 0.4), "peak_margin_mw": (0, 3500),
    "flex_share": (0, 1), "peak_curtailment": (0, 1), "btm_storage_mw": (0, 800),
}

# ---------------------------------------------------------------- scripted demonstration library
# (persona, category) -> proposal. Effect sizes, feasibility and cost are SIMULATED placeholders.
SCRIPTED = {
    ("Grid Engineer", "Interconnection throughput"): dict(
        title="Standardized cluster studies",
        description="Study queued loads in clusters using standardized dynamic models and pre-approved substation designs so each engineer clears more megawatts.",
        ops=(("mul", "ic_mw_per_engineer_year", 1.5),), feasibility=0.70, cost_musd=15),
    ("Energy Economist", "Interconnection throughput"): dict(
        title="Escalating refundable queue deposits",
        description="Require escalating refundable deposits so speculative requests exit the queue and study capacity goes to viable projects.",
        ops=(("add", "ic_bonus_mw", 150.0),), feasibility=0.60, cost_musd=2),
    ("Contrarian Researcher", "Interconnection throughput"): dict(
        title="Flexible-interconnection fast lane",
        description="Offer data centers a pre-studied non-firm connection with automated curtailment instead of a full firm-service study, so flexible load skips the queue.",
        ops=(("add", "ic_bonus_mw", 250.0), ("max", "flex_share", 0.45)), feasibility=0.45, cost_musd=8),

    ("Grid Engineer", "Peak demand"): dict(
        title="Contract fast-start peaking capacity",
        description="Procure new fast-start gas peaking generation to add firm capacity at the system peak.",
        ops=(("add", "peak_margin_mw", 250.0),), feasibility=0.55, cost_musd=300),
    ("Energy Economist", "Peak demand"): dict(
        title="Real-time-price large-load tariff",
        description="Bill large loads at real-time locational prices during peak hours so data centers curtail themselves when the system is tight.",
        ops=(("max", "peak_curtailment", 0.80),), feasibility=0.60, cost_musd=5),
    ("Contrarian Researcher", "Peak demand"): dict(
        title="Data center as virtual power plant",
        description="Pre-cool halls, shift checkpointed training and dispatch on-site batteries so each campus bids negative load into the capacity market.",
        ops=(("max", "peak_curtailment", 0.90), ("max", "flex_share", 0.55), ("add", "btm_storage_mw", 100.0)),
        feasibility=0.45, cost_musd=60),

    ("Grid Engineer", "Transformer availability"): dict(
        title="Statewide standard transformer ratings",
        description="Standardize substation transformer ratings and impedances statewide so spares become interchangeable across utilities.",
        ops=(("mul", "xfmr_existing_multiplier", 1.15),), feasibility=0.65, cost_musd=20),
    ("Energy Economist", "Transformer availability"): dict(
        title="Advance-purchase commitments",
        description="Guarantee manufacturers multi-year advance purchase orders so they expand production lines and commit delivery slots.",
        ops=(("add", "xfmr_units_per_year", 1.5),), feasibility=0.50, cost_musd=90),
    ("Contrarian Researcher", "Transformer availability"): dict(
        title="Medium-voltage DC campuses",
        description="Serve campuses with medium-voltage direct current and solid-state transformers, bypassing grain-oriented steel power transformers entirely.",
        ops=(("add", "xfmr_units_per_year", 1.0), ("mul", "xfmr_existing_multiplier", 1.10)),
        feasibility=0.25, cost_musd=120),

    ("Grid Engineer", "Grid capacity"): dict(
        title="Reconductor congested 115 kV lines",
        description="Replace conductors on congested 115 kV lines with higher-capacity advanced conductors.",
        ops=(("mul", "grid_multiplier", 1.20),), feasibility=0.60, cost_musd=150),
    ("Energy Economist", "Grid capacity"): dict(
        title="Locational headroom pricing",
        description="Publish locational connection charges that steer new campuses toward zones with spare headroom.",
        ops=(("mul", "grid_multiplier", 1.10),), feasibility=0.60, cost_musd=3),
    ("Contrarian Researcher", "Grid capacity"): dict(
        title="Topology optimization + dynamic ratings",
        description="Use software that reconfigures breaker topology and rates lines dynamically to reroute flow around congested elements.",
        ops=(("add", "dlr_uplift", 0.20), ("mul", "grid_multiplier", 1.05)), feasibility=0.60, cost_musd=25),

    ("Grid Engineer", "Flexible workloads"): dict(
        title="Load-shedding relays on DC feeders",
        description="Install under-frequency load-shedding relays on data center feeders so part of the load trips automatically during emergencies.",
        ops=(("max", "peak_curtailment", 0.70),), feasibility=0.70, cost_musd=10),
    ("Energy Economist", "Flexible workloads"): dict(
        title="Interruptible-service discount",
        description="Offer a discounted interruptible rate for training and batch compute that accepts curtailment.",
        ops=(("max", "flex_share", 0.45),), feasibility=0.60, cost_musd=15),
    ("Contrarian Researcher", "Flexible workloads"): dict(
        title="Grid-stress-aware job routing",
        description="Route AI training across regions by grid stress so Massachusetts campuses host only checkpointable work during local peaks.",
        ops=(("max", "flex_share", 0.60), ("max", "peak_curtailment", 0.90)), feasibility=0.40, cost_musd=5),
}


# ---------------------------------------------------------------- LLM proposals (optional)
def llm_available() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _llm_prompt(persona: str, category: str, params: dict, shadow_table: str) -> str:
    plist = "\n".join(f"- {k}: current={params[k]}, allowed range {lo}..{hi}" for k, (lo, hi) in PARAM_BOUNDS.items())
    return f"""You are the {persona}: {PERSONAS[persona]['brief']}
A linear program models deliverable data-center load (MW) in Massachusetts. The binding bottleneck is: {category}.
Constraint shadow prices (MW unlocked per MW relaxed):
{shadow_table}

Propose ONE intervention for this bottleneck in your persona's style. Express its effect ONLY as operations on these model parameters:
{plist}
Allowed op types: "add" (value added), "mul" (multiplier between 0.5 and 2.0), "max" (raise to at least value).
Be conservative and realistic about magnitudes. Do NOT state MW outcomes; the optimizer computes them.
Return ONLY JSON: {{"title": str, "description": str (one sentence), "ops": [{{"op": str, "param": str, "value": number}}], "feasibility": number 0-1, "cost_musd": number, "rationale": str}}"""


def _call_llm(prompt: str, provider: str) -> str:
    if provider == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"},
            json={"model": os.environ.get("UNBOUND_ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
                  "max_tokens": 600, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    r = requests.post("https://api.openai.com/v1/chat/completions", timeout=60, headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": os.environ.get("UNBOUND_OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def validate_proposal(raw: dict) -> dict:
    """Whitelist params/op types and clamp magnitudes. Raises ValueError if unusable."""
    ops = []
    for o in raw.get("ops", [])[:4]:
        op, param, val = o.get("op"), o.get("param"), o.get("value")
        if op not in ("add", "mul", "max") or param not in PARAM_BOUNDS:
            continue
        val = float(val)
        if op == "mul":
            val = float(np.clip(val, 0.5, 2.0))
        lo, hi = PARAM_BOUNDS[param]
        if op == "add":
            val = float(np.clip(val, -(hi - lo), hi - lo))
        if op == "max":
            val = float(np.clip(val, lo, hi))
        ops.append((op, param, val))
    if not ops:
        raise ValueError("proposal had no valid parameter operations")
    return dict(title=str(raw.get("title", "Untitled"))[:80],
                description=str(raw.get("description", ""))[:400],
                ops=tuple(ops),
                feasibility=float(np.clip(float(raw.get("feasibility", 0.5)), 0, 1)),
                cost_musd=max(0.0, float(raw.get("cost_musd", 0) or 0)),
                rationale=str(raw.get("rationale", ""))[:400])


def llm_proposal(persona, category, params, shadow_table, provider):
    text = _call_llm(_llm_prompt(persona, category, params, shadow_table), provider)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("LLM returned no JSON")
    return validate_proposal(json.loads(m.group(0)))


# ---------------------------------------------------------------- evaluation
def _clamp(p: dict) -> dict:
    for k, (lo, hi) in PARAM_BOUNDS.items():
        p[k] = float(np.clip(p[k], lo, hi))
    return p


def apply_ops(params: dict, ops) -> dict:
    iv = Intervention("tmp", "tmp", "", "", tuple(ops), 0.0, False)
    return _clamp(iv.apply(copy.deepcopy(params)))


def run_tournament(params: dict, rounds: int = 3, mode: str = "scripted",
                   w=(1.0, 1.0, 1.0)) -> list:
    """Each round: find the binding bottleneck, collect one proposal per persona,
    score with the LP, adopt the highest-Divergence proposal that unlocks MW,
    then repeat on the updated system."""
    provider = llm_available() if mode == "llm" else None
    cur = copy.deepcopy(params)
    base0 = solve(cur).total_mw
    pipeline = sum(z["pipeline_mw"] for z in cur["zones"])
    stranded0 = max(pipeline - base0, 1e-9)
    history = []
    for rnd in range(1, rounds + 1):
        sol = solve(cur)
        cat = primary_binding(sol)
        if cat not in {c for _, c in SCRIPTED}:
            history.append(dict(round=rnd, bottleneck=cat, base_mw=sol.total_mw, table=None, winner=None))
            break
        sh = sol.constraints[sol.constraints.shadow_price > 0][["constraint", "shadow_price"]]
        shadow_table = sh.to_string(index=False) if len(sh) else "(none)"
        rows = []
        for persona in PERSONAS:
            source, err = "scripted demo", None
            prop = None
            if provider:
                try:
                    prop = llm_proposal(persona, cat, cur, shadow_table, provider)
                    source = f"LLM ({provider})"
                except Exception as e:  # fall back to scripted, but say so
                    err = f"LLM failed: {e}"
            if prop is None:
                prop = dict(SCRIPTED[(persona, cat)], rationale="")
            new_mw = solve(apply_ops(cur, prop["ops"])).total_mw
            rows.append(dict(persona=persona, title=prop["title"], description=prop["description"],
                             ops=describe_ops(Intervention("x", "x", "", "", prop["ops"], 0, False)),
                             _ops=prop["ops"], mw_unlocked=new_mw - sol.total_mw,
                             cost_musd=prop["cost_musd"], feasibility=prop["feasibility"],
                             source=source, note=err or prop.get("rationale", "")))
        t = pd.DataFrame(rows)
        sem = semantic_novelty(t.description)
        t["novelty"] = sem.novelty.to_numpy()
        t["nearest_conventional"] = sem.nearest_conventional.to_numpy()
        t["impact"] = (t.mw_unlocked.clip(lower=0) / stranded0).clip(upper=1)
        t["usd_per_kw"] = np.where(t.mw_unlocked > 0.5, t.cost_musd * 1e6 / (t.mw_unlocked * 1e3), np.nan)
        t["divergence"] = geometric_divergence(t.novelty, t.impact, t.feasibility, *w)
        t = t.sort_values("divergence", ascending=False).reset_index(drop=True)
        useful = t[t.mw_unlocked > 0.5]
        winner = useful.iloc[0] if len(useful) else None
        history.append(dict(round=rnd, bottleneck=cat, base_mw=sol.total_mw, table=t,
                            winner=None if winner is None else winner.to_dict()))
        if winner is None:
            break
        cur = apply_ops(cur, winner["_ops"])
    history.append(dict(round="final", bottleneck=primary_binding(solve(cur)),
                        base_mw=solve(cur).total_mw, table=None, winner=None))
    return history
