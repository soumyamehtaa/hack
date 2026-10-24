"""UNBOUND · AI-powered energy bottleneck intelligence for Massachusetts data centers."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # reach shared `engine` package

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.data import (DEMAND_SCENARIOS, PRESETS, SIM_LABEL, demand_projection,
                         hourly_profile, make_params)
from engine.divergence import (CONVENTIONAL_BASELINE, TOP_K, divergence_scores, embedding_method,
                               semantic_novelty)
from engine import realdata
from engine.interventions import evaluate
from engine.model import (CATEGORIES, cascade, category_summary, primary_binding, solve,
                          sweep)

st.set_page_config(page_title="UNBOUND · Energy Bottleneck Intelligence", page_icon="◐",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- style
ACCENT = "#2EE6A6"
CAT_COLORS = {
    "Grid capacity": "#4EA8DE",
    "Transformer availability": "#F4A261",
    "Interconnection throughput": "#2EE6A6",
    "Peak demand": "#E76F51",
    "Flexible workloads": "#B388FF",
    "Demand pipeline": "#6B7785",
    "Demand pipeline (fully served)": "#6B7785",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0D131A; border-right: 1px solid #1C2530; }
h1, h2, h3 { letter-spacing: -0.02em; }
.ub-brand { display:flex; align-items:baseline; gap:14px; }
.ub-logo { font-weight:700; font-size:2.1rem; letter-spacing:0.18em; color:#E6EDF3; }
.ub-logo span { color:#2EE6A6; }
.ub-tag { color:#8B98A5; font-size:0.95rem; }
.ub-banner { margin:10px 0 18px; padding:9px 14px; border:1px solid #3A2F12; background:#16130A;
             color:#E9C46A; border-radius:8px; font-size:0.82rem; font-family:'JetBrains Mono', monospace; }
.ub-kpi { background:linear-gradient(180deg,#111821 0%,#0E141B 100%); border:1px solid #1C2530;
          border-radius:12px; padding:16px 18px; height:100%; }
.ub-kpi .l { color:#8B98A5; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; }
.ub-kpi .v { color:#E6EDF3; font-size:1.65rem; font-weight:600; margin-top:4px; font-variant-numeric: tabular-nums; }
.ub-kpi .s { color:#6B7785; font-size:0.78rem; margin-top:2px; }
.ub-kpi.hl { border-color:#1F5C4A; box-shadow: 0 0 0 1px rgba(46,230,166,0.08) inset; }
.ub-kpi.hl .v { color:#2EE6A6; }
.ub-insight { border-left:3px solid #2EE6A6; background:#0F1A17; padding:12px 16px; border-radius:6px;
              color:#CFE9DF; font-size:0.92rem; margin:6px 0 14px; }
.ub-muted { color:#6B7785; font-size:0.8rem; }
.ub-real { margin:4px 0 14px; padding:9px 14px; border:1px solid #173A52; background:#0A1722;
           color:#8FD3FF; border-radius:8px; font-size:0.82rem; font-family:'JetBrains Mono', monospace; }
.ub-badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.7rem; font-weight:600;
            letter-spacing:0.06em; margin-right:6px; }
.ub-badge.real { background:#0E2A3D; color:#8FD3FF; border:1px solid #1D4A69; }
.ub-badge.sim { background:#2A220C; color:#E9C46A; border:1px solid #4A3B12; }
.ub-badge.demo { background:#251A33; color:#CDB4FF; border:1px solid #42305C; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom:1px solid #1C2530; }
.stTabs [data-baseweb="tab"] { padding: 8px 16px; color:#8B98A5; }
.stTabs [aria-selected="true"] { color:#E6EDF3 !important; }
</style>
""", unsafe_allow_html=True)


def style_fig(fig, height=380, title=None):
    fig.update_layout(
        template="plotly_dark", height=height, title=dict(text=title, font=dict(size=14)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C9D1D9", size=12),
        margin=dict(l=10, r=10, t=44 if title else 16, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#111821", bordercolor="#2A3644"),
    )
    fig.update_xaxes(gridcolor="#18212B", zerolinecolor="#22303D")
    fig.update_yaxes(gridcolor="#18212B", zerolinecolor="#22303D")
    return fig


def kpi(col, label, value, sub="", highlight=False):
    col.markdown(f"<div class='ub-kpi {'hl' if highlight else ''}'><div class='l'>{label}</div>"
                 f"<div class='v'>{value}</div><div class='s'>{sub}</div></div>",
                 unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar: scenario simulator
sb = st.sidebar
sb.markdown("### Scenario simulator")
preset = sb.selectbox("Preset", list(PRESETS), help="Changing preset resets the sliders.")
p = make_params(preset)
K = lambda name: f"{preset}:{name}"  # keyed per preset so presets reset sliders

p["target_year"] = sb.slider("Target year", 2027, 2035, int(p["target_year"]), key=K("ty"))

sb.markdown("**Interconnection**")
p["ic_engineers"] = sb.slider("Study engineers (FTE)", 2, 40, int(p["ic_engineers"]), key=K("eng"))
p["ic_mw_per_engineer_year"] = sb.slider("MW cleared per engineer-year", 5.0, 60.0,
                                         float(p["ic_mw_per_engineer_year"]), 2.5, key=K("mwe"))
sb.markdown("**Transformers**")
p["xfmr_units_per_year"] = sb.slider("Large transformer deliveries / yr", 0.0, 8.0,
                                     float(p["xfmr_units_per_year"]), 0.5, key=K("xu"))
p["xfmr_mva_per_unit"] = sb.slider("MVA per unit", 20.0, 150.0, float(p["xfmr_mva_per_unit"]), 5.0, key=K("mva"))
p["xfmr_existing_multiplier"] = sb.slider("Existing substation headroom ×", 0.5, 2.0,
                                          float(p["xfmr_existing_multiplier"]), 0.05, key=K("xe"))
sb.markdown("**Grid**")
p["grid_multiplier"] = sb.slider("Grid headroom ×", 0.5, 2.5, float(p["grid_multiplier"]), 0.05, key=K("gm"))
p["dlr_uplift"] = sb.slider("Dynamic line rating uplift", 0.0, 0.4, float(p["dlr_uplift"]), 0.01,
                            format="%.2f", key=K("dlr"))
sb.markdown("**Peak demand**")
p["peak_margin_mw"] = sb.slider("Peak margin for new large load (MW)", 200.0, 2500.0,
                                float(p["peak_margin_mw"]), 50.0, key=K("pk"))
p["btm_storage_mw"] = sb.slider("On-site battery storage (MW)", 0.0, 500.0, float(p["btm_storage_mw"]), 10.0, key=K("bt"))
sb.markdown("**Flexible compute**")
p["flex_share"] = sb.slider("Max flexible share of load", 0.0, 1.0, float(p["flex_share"]), 0.05, key=K("fs"))
p["peak_curtailment"] = sb.slider("Flexible load curtailed at peak", 0.0, 1.0,
                                  float(p["peak_curtailment"]), 0.05, key=K("pc"))
sb.markdown(f"<div class='ub-muted'>{SIM_LABEL}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_eval(params_json: str):
    return evaluate(json.loads(params_json))


@st.cache_data(show_spinner=False)
def cached_cascade(params_json: str):
    return cascade(json.loads(params_json))


@st.cache_data(show_spinner=False)
def cached_sweep(params_json: str, key: str, values: tuple):
    return sweep(json.loads(params_json), key, values)


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching EIA-930 demand for ISO-NE MA load zones…")
def load_real_demand():
    return realdata.fetch_eia_demand()


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching ISO-NE real-time LMP reports…")
def load_real_lmp(days: int, end_iso: str | None):
    from datetime import date as _d
    return realdata.fetch_lmp(days, end=_d.fromisoformat(end_iso) if end_iso else None)


def real_demand_df():
    """Uploaded official CSV takes precedence over the auto-fetched one."""
    up = st.session_state.get("upload_demand")
    if up is not None:
        return up, dict(source="Uploaded EIA-930 sub-region CSV", errors=[])
    return load_real_demand()


pj = json.dumps(p, sort_keys=True)
sol = solve(p)
cs = category_summary(sol)
top = primary_binding(sol)
d = sol.derived
pipeline = sum(z["pipeline_mw"] for z in p["zones"])

# ---------------------------------------------------------------- header
st.markdown("<div class='ub-brand'><div class='ub-logo'>UN<span>BOUND</span></div>"
            "<div class='ub-tag'>Energy bottleneck intelligence · Massachusetts data centers</div></div>",
            unsafe_allow_html=True)
st.markdown(f"<div class='ub-banner'>⚠ {SIM_LABEL} Findings illustrate the method, not the real grid.</div>",
            unsafe_allow_html=True)

if not sol.ok:
    st.error("Optimization infeasible for this scenario.")
    st.stop()

top_shadow = float(cs.loc[cs.category == top, "shadow_price"].sum()) if top in CATEGORIES else 0.0
c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Deliverable DC load", f"{sol.total_mw:,.0f} MW", f"by {p['target_year']} · LP optimum", True)
kpi(c2, "Requested pipeline", f"{pipeline:,.0f} MW", "simulated interconnection requests")
kpi(c3, "Stranded demand", f"{pipeline - sol.total_mw:,.0f} MW",
    f"{(1 - sol.total_mw / pipeline) * 100:.0f}% of pipeline unserved")
kpi(c4, "Binding bottleneck", top.replace(" (fully served)", ""), "highest aggregate shadow price")
kpi(c5, "Shadow price", f"{top_shadow:.2f}", "MW unlocked per MW relaxed")

st.write("")
(T_ENGINE, T_REAL, T_INTERV, T_CASCADE, T_DIV, T_BREAKER, T_DEMAND, T_METHOD) = st.tabs(
    ["Bottleneck engine", "Real-world data", "Interventions", "Bottleneck cascade",
     "Divergence score", "Consensus Breaker", "Demand model", "Methodology"])

# ================================================================ 1. bottleneck engine
with T_ENGINE:
    lever = ""
    if top == "Interconnection throughput":
        per_eng = top_shadow * p["ic_mw_per_engineer_year"] * d["years"]
        lever = f" In this scenario, one additional interconnection engineer is worth ≈ <b>{per_eng:,.0f} MW</b> of deliverable load by {p['target_year']}."
    elif top == "Transformer availability":
        per_unit = top_shadow * p["xfmr_mva_per_unit"] * p["power_factor"] * d["years"]
        lever = f" One more large transformer per year is worth up to ≈ <b>{per_unit:,.0f} MW</b> by {p['target_year']} (if it lands where it binds)."
    elif top == "Peak demand":
        lever = " Here, firm capacity at system peak binds; flexible workloads and storage compete directly with new generation."
    st.markdown(f"<div class='ub-insight'>The optimizer serves <b>{sol.total_mw:,.0f} MW</b>; the binding constraint is "
                f"<b>{top}</b>.{lever}</div>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        cdf = sol.constraints.copy()
        cdf = cdf[cdf.category != "Demand pipeline"].sort_values("shadow_price", ascending=True).tail(12)
        fig = go.Figure(go.Bar(
            x=cdf.shadow_price, y=cdf.constraint, orientation="h",
            marker_color=[CAT_COLORS[c] for c in cdf.category],
            text=[f"{v:.2f}" if v > 0 else "slack" for v in cdf.shadow_price], textposition="outside",
            hovertemplate="%{y}<br>shadow price %{x:.3f} MW/MW<extra></extra>"))
        style_fig(fig, 420, "Shadow prices · MW of DC load unlocked per MW of constraint relaxed")
        fig.update_xaxes(range=[0, max(1.2, cdf.shadow_price.max() * 1.25)])
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
    with right:
        cdf = sol.constraints
        heat_cats = ["Grid capacity", "Transformer availability", "Flexible workloads", "Demand pipeline"]
        z = []
        for zn in sol.zones:
            row = []
            for cat in heat_cats:
                v = cdf[(cdf.zone == zn) & (cdf.category == cat)].utilization
                row.append(float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else np.nan)
            z.append(row)
        fig = go.Figure(go.Heatmap(
            z=z, x=[c.replace(" availability", "") for c in heat_cats], y=sol.zones,
            colorscale=[[0, "#0E1A22"], [0.6, "#1F5C4A"], [0.95, "#2EE6A6"], [1, "#F4F1BB"]],
            zmin=0, zmax=1, text=[[f"{v:.0%}" if pd.notna(v) else "–" for v in r] for r in z],
            texttemplate="%{text}", colorbar=dict(title="util.", thickness=10)))
        style_fig(fig, 260, "Zone utilization of local constraints")
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

        sw = cdf[cdf.zone == "Statewide"]
        fig = go.Figure()
        for _, r in sw.iterrows():
            fig.add_trace(go.Bar(x=[r.utilization], y=[r.constraint.split(" (")[0]], orientation="h",
                                 marker_color=CAT_COLORS[r.category], showlegend=False,
                                 text=f"{r.lhs:,.0f} / {r.rhs:,.0f} MW", textposition="inside",
                                 hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        fig.update_xaxes(range=[0, 1.05], tickformat=".0%")
        style_fig(fig, 170, "Statewide constraint utilization")
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

    fig = go.Figure()
    fig.add_trace(go.Bar(x=sol.zones, y=sol.firm, name="Firm load served", marker_color="#4EA8DE"))
    fig.add_trace(go.Bar(x=sol.zones, y=sol.flex, name="Flexible load served", marker_color="#B388FF"))
    fig.add_trace(go.Scatter(x=sol.zones, y=[z["pipeline_mw"] for z in p["zones"]], name="Requested pipeline",
                             mode="markers", marker=dict(symbol="line-ew-open", size=40, color="#E9C46A",
                                                         line=dict(width=3))))
    fig.update_layout(barmode="stack")
    style_fig(fig, 320, "Optimal allocation by zone (MW)")
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

    with st.expander("Full constraint table"):
        show = sol.constraints[["constraint", "category", "rhs", "lhs", "slack", "shadow_price", "utilization", "binding"]]
        st.dataframe(show.round(3), width="stretch", hide_index=True)

# ================================================================ real-world data
with T_REAL:
    st.markdown("<div class='ub-real'>● REAL OBSERVATIONS: hourly demand for ISO-NE's three Massachusetts load "
                "zones (SEMASS, WCMASS, NEMASSBOST) from U.S. EIA Form EIA-930, and real-time hourly LMPs from "
                "ISO New England. Nothing on this tab is simulated except where marked "
                "<span class='ub-badge sim'>SIM</span>.</div>", unsafe_allow_html=True)

    with st.expander("Import official CSV instead of auto-fetch"):
        st.caption("Accepted: EIA-930 SUBREGION six-month file (eia.gov/electricity/gridmonitor) or an ISO-NE hourly "
                   "LMP report (RT/DA, final or prelim, from ISO Express). Uploads override fetched data for this session.")
        ups = st.file_uploader("Official CSV files", type=["csv"], accept_multiple_files=True, key="real_upload")
        st.session_state["upload_demand"], st.session_state["upload_lmp"] = None, None
        dem_parts, lmp_parts = [], []
        for f in ups or []:
            try:
                kind, df_up = realdata.parse_upload(f.name, f.getvalue())
                (dem_parts if kind == "demand" else lmp_parts).append(df_up)
                st.success(f"{f.name}: parsed {len(df_up):,} {kind} rows")
            except Exception as e:
                st.error(str(e))
        if dem_parts:
            st.session_state["upload_demand"] = pd.concat(dem_parts).drop_duplicates(["ts", "zone"]).sort_values("ts")
        if lmp_parts:
            st.session_state["upload_lmp"] = pd.concat(lmp_parts).sort_values("ts")

    dem, dmeta = real_demand_df()
    if dem is None:
        st.error("Could not retrieve EIA-930 demand data and no upload provided. "
                 + "; ".join(dmeta.get("errors", [])))
    else:
        ds = realdata.demand_stats(dem)
        r1, r2, r3, r4, r5 = st.columns(5)
        kpi(r1, "MA observed peak", f"{ds['peak_mw']:,.0f} MW", f"{ds['peak_ts']:%b %d %Y, HE %H}", True)
        kpi(r2, "MA mean load", f"{ds['mean_mw']:,.0f} MW", f"{ds['start']:%b %d} – {ds['end']:%b %d %Y}")
        kpi(r3, "Load factor", f"{ds['load_factor']:.0%}", "mean ÷ peak, observed")
        kpi(r4, "Hours ≥95% of peak", f"{ds['hours_within_5pct']}", f"of {ds['hours']:,} complete hours")
        kpi(r5, "Simulated DC load vs. peak", f"{sol.total_mw / ds['peak_mw']:.1%}",
            "SIM LP optimum ÷ REAL peak")

        ma = realdata.ma_hourly(dem)
        zw = dem.pivot_table(index="ts", columns="zone", values="demand_mw").loc[ma.index]
        fig = go.Figure()
        for zn, colr in zip(["NEMASSBOST", "SEMASS", "WCMASS"], ["#4EA8DE", "#2EE6A6", "#B388FF"]):
            if zn in zw:
                fig.add_trace(go.Scatter(x=zw.index, y=zw[zn], stackgroup="z", name=zn,
                                         line=dict(width=0.4, color=colr)))
        fig.add_hline(y=ds["peak_mw"] * 0.95, line=dict(color="#E76F51", dash="dot"),
                      annotation_text="95% of observed peak", annotation_font_color="#E76F51")
        style_fig(fig, 340, "Massachusetts hourly demand by ISO-NE load zone (MW) · EIA-930")
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

        a, b = st.columns(2)
        with a:
            ldc = np.sort(ma.to_numpy())[::-1]
            fig = go.Figure(go.Scatter(x=np.arange(1, len(ldc) + 1), y=ldc, mode="lines",
                                       line=dict(color="#4EA8DE", width=2), name="observed"))
            fig.add_vrect(x0=0, x1=ds["hours_within_10pct"], fillcolor="rgba(231,111,81,0.12)", line_width=0,
                          annotation_text=f"{ds['hours_within_10pct']} h within 10% of peak",
                          annotation_font_color="#E76F51", annotation_position="top right")
            style_fig(fig, 340, "Load-duration curve (observed MA demand)")
            fig.update_xaxes(title="hours, sorted", type="log")
            fig.update_yaxes(title="MW")
            st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
        with b:
            xs = np.linspace(100, 3000, 30)
            hc = realdata.flexible_headroom_curve(dem, xs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hc.added_mw, y=hc.hours_over_peak, name="hours above observed peak",
                                     line=dict(color="#E76F51", width=2)))
            fig.add_trace(go.Scatter(x=hc.added_mw, y=hc.curtailed_share * 100, name="% of new load's energy to shift",
                                     line=dict(color="#2EE6A6", width=2), yaxis="y2"))
            fig.add_vline(x=sol.total_mw, line=dict(color="#E9C46A", dash="dot"),
                          annotation_text="SIM LP optimum", annotation_font_color="#E9C46A")
            style_fig(fig, 340, "If new flat load had to fit under the observed peak")
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="% energy", gridcolor="rgba(0,0,0,0)"))
            fig.update_xaxes(title="new flat data-center load (MW)")
            fig.update_yaxes(title="hours")
            st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
        row = hc.iloc[(hc.added_mw - 1000).abs().argmin()]
        st.markdown(f"<div class='ub-insight'>Computed on observed data: a flat <b>{row.added_mw:,.0f} MW</b> block "
                    f"would push MA zonal demand above its observed {ds['start']:%Y} peak in <b>{row.hours_over_peak:.0f} "
                    f"hours</b>; keeping under that peak would require shifting <b>{row.curtailed_share:.2%}</b> of the "
                    f"block's energy. This treats the observed peak as a ceiling, which is an assumption "
                    f"(real limits depend on transmission, reserves and imports). It is the empirical case for "
                    f"the model's flexible-workload lever.</div>", unsafe_allow_html=True)

        # ---- prices
        st.markdown("##### Real-time prices · MA load zones")
        pc1, pc2 = st.columns([1, 3])
        window = pc1.radio("Window", ["Last 14 days", "Week of observed peak"], key="lmp_window")
        up_lmp = st.session_state.get("upload_lmp")
        if up_lmp is not None:
            lmp, lmeta = up_lmp, dict(source="Uploaded ISO-NE LMP CSV", days=up_lmp.ts.dt.date.nunique(), prelim_days=0)
        elif window == "Last 14 days":
            lmp, lmeta = load_real_lmp(14, None)
        else:
            lmp, lmeta = load_real_lmp(7, (ds["peak_ts"] + pd.Timedelta(days=3)).date().isoformat())
        if lmp is None:
            pc2.error("Could not retrieve ISO-NE LMP reports; upload one above.")
        else:
            shape, _ = realdata.diurnal_shape(dem)
            pw = realdata.peak_window(shape)
            sp = realdata.peak_vs_offpeak_price(lmp, pw)
            ls = realdata.lmp_stats(lmp)
            q1, q2, q3 = pc2.columns(3)
            kpi(q1, "Mean MA LMP", f"${ls[ls.location != 'INTERNAL_HUB'].mean_lmp.mean():,.2f}", "$/MWh, RT")
            kpi(q2, f"Peak-window LMP (HB {pw[0]}–{pw[-1]})", f"${sp['peak']:,.2f}",
                f"vs ${sp['offpeak']:,.2f} off-window")
            kpi(q3, "Max hourly LMP", f"${ls.max_lmp.max():,.2f}", f"{int(ls.hours_over_100.max())} h > $100")
            fig = go.Figure()
            for loc, colr in zip(["NEMASSBOST", "SEMASS", "WCMASS", "INTERNAL_HUB"],
                                 ["#4EA8DE", "#2EE6A6", "#B388FF", "#6B7785"]):
                g = lmp[lmp.location == loc]
                fig.add_trace(go.Scatter(x=g.ts, y=g.lmp, name=loc, line=dict(color=colr, width=1.4,
                                                                               dash="dot" if loc == "INTERNAL_HUB" else None)))
            style_fig(fig, 320, f"Real-time LMP ($/MWh) · {lmeta.get('days')} daily ISO-NE reports"
                                + (f" ({lmeta['prelim_days']} preliminary)" if lmeta.get("prelim_days") else ""))
            st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
            st.dataframe(ls.round(2), width="stretch", hide_index=True)
            st.caption(f"Peak window = 5 hours around the hour of maximum load on the 10 highest-load days in the "
                       f"demand record (observed: HB {pw[0]}–{pw[-1]}). Mean congestion component near zero means "
                       f"MA zones priced close to the hub in this window.")

        st.markdown("<div class='ub-muted'>Sources: "
                    "<a href='https://www.eia.gov/electricity/gridmonitor/' target='_blank'>EIA Hourly Electric Grid "
                    "Monitor (EIA-930)</a> · <a href='https://www.iso-ne.com/isoexpress/web/reports/pricing/-/tree/lmps-rt-hourly-final' "
                    f"target='_blank'>ISO-NE hourly RT LMP</a>. Demand source: {dmeta.get('source')}. "
                    f"Retrieved {dmeta.get('retrieved', 'n/a')}. EIA-930 data are operator-reported and may be revised. "
                    "Infrastructure capacities in the optimizer remain SIMULATED.</div>", unsafe_allow_html=True)

# ================================================================ 2. interventions
with T_INTERV:
    with st.spinner("Re-solving LP for every intervention subset ordering…"):
        ev = cached_eval(pj)
    basis = st.radio("Impact basis", ["Shapley (interaction-aware)", "Standalone"], horizontal=True,
                     help="Standalone: MW gained applying only this intervention. Shapley: average marginal MW "
                          "across 40 random orderings of all interventions, which credits levers that only "
                          "pay off once another bottleneck is cleared.")
    col = "shapley_mw" if basis.startswith("Shapley") else "standalone_mw"
    evs = ev.sort_values(col, ascending=True)
    best = evs.iloc[-1]
    st.markdown(f"<div class='ub-insight'>Top lever: <b>{best.intervention}</b> unlocks "
                f"<b>{best[col]:,.0f} MW</b> ({basis.split(' ')[0].lower()}). Applying all {len(ev)} interventions "
                f"moves deliverable load from {ev.attrs['base_mw']:,.0f} → <b>{ev.attrs['all_mw']:,.0f} MW</b>.</div>",
                unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=evs[col], y=evs.intervention, orientation="h",
        marker=dict(color=[CAT_COLORS[c] for c in evs.category],
                    line=dict(color=["#E6EDF3" if c else "rgba(0,0,0,0)" for c in evs.consensus], width=1)),
        text=[f"{v:,.0f} MW" for v in evs[col]], textposition="outside",
        customdata=np.stack([evs.category, evs.effect], axis=1),
        hovertemplate="%{y}<br>%{x:,.1f} MW<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    style_fig(fig, 520, "Additional deliverable MW by intervention  (white outline = consensus lever)")
    fig.update_xaxes(range=[min(0, evs[col].min()), evs[col].max() * 1.2 + 1])
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
    st.dataframe(
        ev.sort_values(col, ascending=False)[["intervention", "category", "standalone_mw", "shapley_mw",
                                              "feasibility", "consensus", "effect"]].round(1),
        width="stretch", hide_index=True,
        column_config={"standalone_mw": "Standalone MW", "shapley_mw": "Shapley MW",
                       "feasibility": st.column_config.ProgressColumn("Feasibility (sim.)", min_value=0, max_value=1)})
    st.markdown("<div class='ub-muted'>Effect sizes and feasibility are simulated priors, not engineering estimates.</div>",
                unsafe_allow_html=True)

# ================================================================ 3. cascade
with T_CASCADE:
    steps = cached_cascade(pj)
    chain = " → ".join([s["exposed"] for s in steps if s["exposed"]])
    st.markdown(f"<div class='ub-insight'>Removing one bottleneck exposes the next: <b>{chain}</b>.</div>",
                unsafe_allow_html=True)
    labels = [s["action"].replace("Remove ", "− ") for s in steps]
    mw = [s["total_mw"] for s in steps]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[mw[0]] + [mw[i] - mw[i - 1] for i in range(1, len(mw))],
                         base=[0] + mw[:-1],
                         marker_color=["#2A3644"] + [CAT_COLORS[s["action"].replace("Remove ", "")] for s in steps[1:]],
                         text=[f"{mw[0]:,.0f}"] + [f"+{mw[i] - mw[i - 1]:,.0f}" for i in range(1, len(mw))],
                         textposition="outside", showlegend=False,
                         hovertemplate="%{x}<br>%{text} MW<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=mw, mode="lines+markers", line=dict(color=ACCENT, width=2, dash="dot"),
                             name="Deliverable MW", marker=dict(size=9)))
    for i, s in enumerate(steps):
        if s["exposed"]:
            fig.add_annotation(x=labels[i], y=mw[i], yshift=-28 if i else 28, showarrow=False,
                               text=f"binds: {s['exposed']}", font=dict(size=10, color=CAT_COLORS.get(s["exposed"], "#aaa")))
    fig.add_hline(y=pipeline, line=dict(color="#E9C46A", dash="dash"), annotation_text="requested pipeline",
                  annotation_font_color="#E9C46A")
    style_fig(fig, 440, "Bottleneck cascade · sequentially removing the binding constraint")
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

    st.markdown("##### Relaxation sweep")
    st.caption("Sweep one lever and watch the binding constraint switch. Flat segments mean the lever has stopped mattering.")
    LEVERS = {
        "Interconnection engineers": ("ic_engineers", np.linspace(2, 40, 39)),
        "Transformer deliveries / yr": ("xfmr_units_per_year", np.linspace(0, 8, 33)),
        "Peak margin (MW)": ("peak_margin_mw", np.linspace(200, 2500, 47)),
        "Flexible load curtailed at peak": ("peak_curtailment", np.linspace(0, 1, 41)),
        "Grid headroom ×": ("grid_multiplier", np.linspace(0.5, 2.5, 41)),
        "On-site battery storage (MW)": ("btm_storage_mw", np.linspace(0, 500, 41)),
    }
    lv = st.selectbox("Lever", list(LEVERS), key="sweep_lever")
    key, vals = LEVERS[lv]
    sw = cached_sweep(pj, key, tuple(np.round(vals, 4)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sw.value, y=sw.total_mw, mode="lines", line=dict(color="#3A4756", width=2),
                             showlegend=False, hoverinfo="skip"))
    for cat, g in sw.groupby("binding"):
        fig.add_trace(go.Scatter(x=g.value, y=g.total_mw, mode="markers", name=f"binding: {cat}",
                                 marker=dict(color=CAT_COLORS.get(cat, "#999"), size=8),
                                 hovertemplate=f"{lv}=%{{x}}<br>%{{y:,.0f}} MW<br>{cat}<extra></extra>"))
    fig.add_vline(x=p[key], line=dict(color="#E6EDF3", dash="dot"), annotation_text="current")
    style_fig(fig, 380, f"Deliverable MW vs. {lv}")
    fig.update_xaxes(title=lv)
    fig.update_yaxes(title="MW")
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

# ================================================================ 4. divergence
with T_DIV:
    st.markdown("<div class='ub-insight'>The Divergence Score surfaces interventions that are far from the mainstream "
                "“build more generation” narrative <i>and</i> still move MW <i>and</i> are buildable. "
                "Weighted geometric mean: an idea must score on all three axes.</div>", unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    wn = w1.slider("Weight · semantic novelty", 0.0, 3.0, 1.0, 0.1)
    wi = w2.slider("Weight · energy impact", 0.0, 3.0, 1.0, 0.1)
    wf = w3.slider("Weight · feasibility", 0.0, 3.0, 1.0, 0.1)
    ib = w4.radio("Impact basis", ["Shapley", "Standalone"], key="div_basis")
    ev = cached_eval(pj)
    dv = divergence_scores(ev, "shapley_mw" if ib == "Shapley" else "standalone_mw", wn, wi, wf)

    a, b = st.columns([1.2, 1])
    with a:
        fig = go.Figure(go.Scatter(
            x=dv.novelty, y=dv.impact, mode="markers+text", text=dv.intervention, textposition="top center",
            textfont=dict(size=10, color="#AAB4BF"),
            marker=dict(size=12 + 34 * dv.feasibility, color=dv.divergence,
                        colorscale=[[0, "#1A2430"], [0.5, "#1F5C4A"], [1, "#2EE6A6"]],
                        line=dict(color=["#E6EDF3" if c else "#0A0E13" for c in dv.consensus], width=1.5),
                        showscale=True, colorbar=dict(title="Div.", thickness=10)),
            customdata=np.stack([dv.divergence, dv.feasibility], axis=1),
            hovertemplate="%{text}<br>novelty %{x:.2f} · impact %{y:.2f}<br>feasibility %{customdata[1]:.2f}"
                          "<br>divergence %{customdata[0]:.1f}<extra></extra>"))
        fig.add_shape(type="rect", x0=0.5, x1=1.05, y0=0.5, y1=1.08, line=dict(color="#2EE6A6", dash="dot"),
                      fillcolor="rgba(46,230,166,0.04)")
        fig.add_annotation(x=1.03, y=1.06, text="out-of-consensus & high-impact", showarrow=False,
                           xanchor="right", font=dict(color=ACCENT, size=11))
        style_fig(fig, 520, "Novelty × impact  (bubble = feasibility, white outline = consensus)")
        fig.update_xaxes(title="Semantic distance from conventional baseline", range=[-0.05, 1.08])
        fig.update_yaxes(title="Normalized MW impact", range=[-0.08, 1.12])
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
    with b:
        st.dataframe(dv[["intervention", "divergence", "novelty", "impact", "feasibility",
                         "nearest_conventional"]].round(2),
                     width="stretch", hide_index=True, height=520,
                     column_config={"divergence": st.column_config.ProgressColumn("Divergence", min_value=0,
                                                                                  max_value=100, format="%.0f"),
                                    "impact": "Impact (sim., norm.)", "feasibility": "Feasibility (sim.)",
                                    "nearest_conventional": "Nearest conventional idea"})
    st.caption(f"Novelty engine: **{embedding_method()}** · novelty = 1 − mean of top-{TOP_K} cosine "
               f"similarities to {len(CONVENTIONAL_BASELINE)} conventional interventions (absolute scale, not rescaled).")

    st.markdown("##### Test a hypothesis")
    hyp = st.text_area("Describe an energy intervention", key="hyp_text",
                       placeholder="e.g. Treat data centers as virtual power plants that sell curtailment back to ISO-NE")
    if hyp.strip():
        r = semantic_novelty([hyp.strip()]).iloc[0]
        h1, h2 = st.columns([1, 3])
        kpi(h1, "Semantic novelty", f"{r.novelty:.2f}", "0 = conventional · 1 = unrelated", r.novelty >= 0.6)
        h2.markdown(f"<div class='ub-insight'>Closest conventional idea (cos {r.nearest_similarity:.2f}): "
                    f"<i>{r.nearest_conventional}</i><br><span class='ub-muted'>Novelty only. MW impact requires "
                    f"mapping the idea to model parameters; see Consensus Breaker.</span></div>",
                    unsafe_allow_html=True)
    with st.expander("Conventional-intervention baseline (hand-written reference set)"):
        for s in CONVENTIONAL_BASELINE:
            st.markdown(f"- {s}")

# ================================================================ 5. demand model
with T_DEMAND:
    st.markdown("<div class='ub-insight'>Simulated statewide data-center demand trajectories versus what the "
                "optimizer says the system can actually deliver by the target year.</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        proj = demand_projection()
        fig = go.Figure()
        palette = ["#4EA8DE", "#B388FF", "#E76F51"]
        for (name, (yrs, mwv)), colr in zip(proj.items(), palette):
            fig.add_trace(go.Scatter(x=yrs, y=mwv, mode="lines", name=f"{name} ({DEMAND_SCENARIOS[name]:.0%}/yr)",
                                     line=dict(color=colr, width=2)))
        fig.add_trace(go.Scatter(x=[p["target_year"]], y=[sol.total_mw], mode="markers+text",
                                 marker=dict(color=ACCENT, size=14, symbol="diamond"),
                                 text=[f"deliverable {sol.total_mw:,.0f} MW"], textposition="middle left",
                                 name="LP deliverable"))
        style_fig(fig, 400, "Projected MA data-center load (simulated, MW)")
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
    with b:
        firm, flex = float(sol.firm.sum()), float(sol.flex.sum())
        _dem, _ = real_demand_df()
        if _dem is not None:
            real_shape, _ = realdata.diurnal_shape(_dem)
            hrs, f_h, x_h, _, pk = hourly_profile(firm, flex, p["peak_curtailment"],
                                                  tuple(realdata.peak_window(real_shape)))
            sys_shape, shape_label = real_shape, "REAL MA load shape, top-10 days (norm.)"
        else:
            hrs, f_h, x_h, sys_shape, pk = hourly_profile(firm, flex, p["peak_curtailment"])
            shape_label = "SIM system load shape (norm.)"
        fig = go.Figure()
        fig.add_vrect(x0=min(pk) - 0.5, x1=max(pk) + 0.5, fillcolor="rgba(231,111,81,0.10)", line_width=0,
                      annotation_text="system peak window", annotation_font_color="#E76F51")
        fig.add_trace(go.Scatter(x=hrs, y=f_h, stackgroup="dc", name="Firm", line=dict(color="#4EA8DE", width=0.5)))
        fig.add_trace(go.Scatter(x=hrs, y=x_h, stackgroup="dc", name="Flexible (shifted)",
                                 line=dict(color="#B388FF", width=0.5)))
        fig.add_trace(go.Scatter(x=hrs, y=sys_shape * (firm + flex) * 1.1, name=shape_label,
                                 line=dict(color="#6B7785", dash="dot"), yaxis="y"))
        style_fig(fig, 400, "Optimal 24h data-center load profile (MW)")
        fig.update_xaxes(title="hour", dtick=2)
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
    years_needed = {}
    for name, (yrs, mwv) in demand_projection(2024, 2045).items():
        hit = yrs[mwv >= sol.total_mw]
        years_needed[name] = int(hit[0]) if len(hit) else None
    st.markdown(" · ".join(f"**{n}** reaches the deliverable ceiling in **{y or '>2045'}**"
                           for n, y in years_needed.items()))

# ================================================================ 6. methodology
with T_METHOD:
    st.markdown("#### Optimization model")
    st.markdown("A linear program solved with SciPy `linprog` (HiGHS). Per zone *i*: firm load $f_i$, "
                "flexible load $s_i$, new transformer allocation $n_i$.")
    st.latex(r"\max \sum_i (f_i + s_i)")
    st.latex(r"f_i + (1-c)\,s_i \le G_i(1+\delta)\,m_g + \beta_i \quad\text{(grid)}")
    st.latex(r"f_i + s_i - n_i \le E_i,\qquad \textstyle\sum_i n_i \le u\,T\,\text{MVA}\,\text{pf} \quad\text{(transformers)}")
    st.latex(r"\textstyle\sum_i (f_i + s_i) \le e\,r\,T \quad\text{(interconnection throughput)}")
    st.latex(r"\textstyle\sum_i f_i + (1-c)\,s_i \le P + B \quad\text{(peak)}")
    st.latex(r"(1-\phi)\,s_i - \phi f_i \le 0,\qquad f_i + s_i \le D_i")
    st.markdown("""
- **Shadow prices** are the LP duals: MW of additional data-center load per unit of constraint relaxed. They are local; large changes are evaluated by re-solving.
- **Cascade**: repeatedly remove the binding category that unlocks the most MW and record what binds next.
- **Shapley impact**: mean marginal MW of each intervention across 40 random orderings of all interventions, which handles bottleneck interactions.
- **Semantic novelty** = 1 − mean top-3 cosine similarity between sentence embeddings (all-MiniLM-L6-v2) of an idea and a baseline of conventional interventions.
- **Divergence** = 100 · weighted geometric mean(novelty, normalized impact, feasibility).
""")
    st.markdown("#### Data status")
    st.warning(SIM_LABEL + " Zone headroom, pipeline, lead times, labor productivity, effect sizes and feasibility "
               "scores are placeholders. Candidate real sources to integrate: ISO-NE CELT forecasts and "
               "interconnection queue, utility hosting-capacity maps, EIA-860/861, LBNL Queued Up, "
               "transformer lead-time surveys, BLS labor data.")
