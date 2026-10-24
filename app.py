"""DIVERGE · human out-of-consensus reasoning -> high-information AI training data.

(The earlier UNBOUND grid app is preserved at unbound/unbound_app.py.)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from diverge import experiment as X
from diverge import store
from diverge.baselines import consensus_assumptions, get_baselines
from diverge.questions import BY_ID, QUESTIONS
from diverge.scoring import CLAIM_NEW_SIM, VERSION, WEIGHTS, score_submission

st.set_page_config(page_title="DIVERGE · Human Intelligence Exchange", page_icon="◇", layout="wide",
                   initial_sidebar_state="collapsed")

PURPLE, VIOLET, LIME, INK, MUTED = "#8B5CF6", "#A78BFA", "#C6FF3D", "#ECEAF5", "#8A85A3"
ARM_COLORS = {"divergent": LIME, "conventional": PURPLE, "random": "#5B5870"}
COMP_LABEL = {"semantic_divergence": "Semantic divergence", "conceptual_novelty": "Conceptual novelty",
              "reasoning_quality": "Reasoning quality", "evidence_quality": "Evidence quality",
              "information_gain": "Information gain"}
PLOT_CFG = {"displayModeBar": False}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown {{ font-family:'Inter',sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,0.18), transparent 60%),
                       radial-gradient(900px 500px at -10% 10%, rgba(198,255,61,0.06), transparent 60%), #07060C; }}
.block-container {{ padding-top:1.4rem; max-width:1380px; }}
h1,h2,h3,h4,h5 {{ font-family:'Space Grotesk',sans-serif; letter-spacing:-0.02em; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.dv-nav {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }}
.dv-logo {{ font-family:'Space Grotesk'; font-weight:700; font-size:2.0rem; letter-spacing:0.22em; color:{INK}; }}
.dv-logo b {{ color:{LIME}; font-weight:700; }}
.dv-tag {{ color:{MUTED}; font-size:0.92rem; }}
.dv-pill {{ display:inline-block; font-family:'JetBrains Mono'; font-size:0.7rem; padding:3px 10px; border-radius:999px;
           border:1px solid #2B2445; color:{VIOLET}; background:#110D1E; margin-left:6px; }}
.dv-pill.lime {{ color:{LIME}; border-color:#3A4A12; background:#10140A; }}
.dv-pill.warn {{ color:#FFC857; border-color:#4A3A12; background:#16120A; }}
.dv-hero {{ border:1px solid #221C38; border-radius:18px; padding:22px 26px; margin:8px 0 18px;
           background:linear-gradient(135deg, rgba(139,92,246,0.12), rgba(18,15,31,0.6) 55%, rgba(198,255,61,0.05)); }}
.dv-hero h2 {{ margin:0 0 6px; font-size:1.55rem; }}
.dv-hero p {{ color:#B9B4CF; margin:0; font-size:0.95rem; }}
.dv-card {{ border:1px solid #221C38; background:rgba(18,15,31,0.75); border-radius:14px; padding:16px 18px; height:100%; }}
.dv-card .l {{ color:{MUTED}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.14em; }}
.dv-card .v {{ font-family:'Space Grotesk'; font-size:1.7rem; font-weight:600; color:{INK}; margin-top:2px; }}
.dv-card .s {{ color:#6E6987; font-size:0.78rem; }}
.dv-card.hl .v {{ color:{LIME}; }}
.dv-q {{ font-family:'Space Grotesk'; font-size:1.35rem; line-height:1.35; color:{INK}; margin:4px 0 2px; }}
.dv-base {{ border-left:2px solid {PURPLE}; background:#0F0C1A; padding:10px 14px; border-radius:6px; margin-bottom:8px;
           color:#BDB8D3; font-size:0.86rem; }}
.dv-assume {{ border:1px solid #2B2445; border-radius:10px; padding:10px 14px; margin-bottom:8px; background:#0E0B18; }}
.dv-assume .t {{ color:{INK}; font-size:0.9rem; }}
.dv-assume .m {{ font-family:'JetBrains Mono'; font-size:0.72rem; color:{MUTED}; margin-top:4px; }}
.dv-ok {{ color:{LIME}; }} .dv-no {{ color:#FF6B8B; }} .dv-mid {{ color:#FFC857; }}
.dv-insight {{ border-left:3px solid {LIME}; background:#0E1208; padding:12px 16px; border-radius:6px; color:#DDE8C4;
              font-size:0.92rem; margin:6px 0 14px; }}
.dv-measured {{ border:1px solid #3A4A12; background:#0D1108; color:#D8F59A; border-radius:10px; padding:10px 14px;
               font-family:'JetBrains Mono'; font-size:0.78rem; margin-bottom:10px; }}
.dv-hypo {{ border:1px solid #4A3A12; background:#15110A; color:#FFD98A; border-radius:10px; padding:10px 14px;
           font-family:'JetBrains Mono'; font-size:0.78rem; margin-bottom:10px; }}
.dv-muted {{ color:#6E6987; font-size:0.8rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid #221C38; }}
.stTabs [data-baseweb="tab"] {{ padding:10px 16px; color:{MUTED}; font-family:'Space Grotesk'; }}
.stTabs [aria-selected="true"] {{ color:{INK} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{LIME}; }}
.stButton > button[kind="primary"], .stDownloadButton > button {{ background:{LIME}; color:#0A0A0A; border:none; font-weight:600; }}
.stTextArea textarea, .stTextInput input {{ background:#0E0B18 !important; border:1px solid #2B2445 !important; }}
</style>""", unsafe_allow_html=True)


def card(col, label, value, sub="", hl=False):
    col.markdown(f"<div class='dv-card {'hl' if hl else ''}'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def style(fig, h=360, title=None):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color="#CFCBE3", size=12),
                      title=dict(text=title, font=dict(family="Space Grotesk", size=15)) if title else None,
                      margin=dict(l=10, r=10, t=46 if title else 14, b=10),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#120F1F", bordercolor="#2B2445"))
    fig.update_xaxes(gridcolor="#1A1530", zerolinecolor="#241E3B")
    fig.update_yaxes(gridcolor="#1A1530", zerolinecolor="#241E3B")
    return fig


def gauge(score: float, h=250):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(family="Space Grotesk", size=54, color=LIME)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor="#3A3355", tickfont=dict(color=MUTED)), bar=dict(color=LIME, thickness=0.28),
                   bgcolor="#0E0B18", borderwidth=0,
                   steps=[dict(range=[0, 25], color="#140F22"), dict(range=[25, 50], color="#1B1430"),
                          dict(range=[50, 75], color="#25193F"), dict(range=[75, 100], color="#33205A")],
                   threshold=dict(line=dict(color=VIOLET, width=3), value=store.TRAINING_THRESHOLD))))
    return style(fig, h)


def radar(ev, h=330):
    keys = list(WEIGHTS)
    raw = [ev["components"][k] for k in keys]
    cred = [ev["credited"][k] for k in keys]
    labels = [COMP_LABEL[k] for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=raw + raw[:1], theta=labels + labels[:1], name="raw component",
                                  line=dict(color=PURPLE, width=1.5, dash="dot"), fill="toself",
                                  fillcolor="rgba(139,92,246,0.10)"))
    fig.add_trace(go.Scatterpolar(r=cred + cred[:1], theta=labels + labels[:1], name="credited (after quality gate)",
                                  line=dict(color=LIME, width=2), fill="toself", fillcolor="rgba(198,255,61,0.14)"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], gridcolor="#241E3B", showticklabels=False),
                                 angularaxis=dict(gridcolor="#241E3B", tickfont=dict(size=11))))
    return style(fig, h)


def contributions_bar(ev, h=230):
    keys = list(WEIGHTS)
    fig = go.Figure()
    for k, colr in zip(keys, [LIME, "#9BE15D", VIOLET, PURPLE, "#5E3FBF"]):
        fig.add_trace(go.Bar(y=["score"], x=[ev["contributions"][k]], orientation="h", name=f"{COMP_LABEL[k]} ({WEIGHTS[k]:.0%})",
                             marker_color=colr, hovertemplate=f"{COMP_LABEL[k]}: %{{x:.1f}} pts<extra></extra>"))
        fig.add_trace(go.Bar(y=["max"], x=[100 * WEIGHTS[k]], orientation="h", marker_color="#1A1530", showlegend=False,
                             marker_line=dict(color=colr, width=1), hoverinfo="skip"))
    fig.update_layout(barmode="stack", legend=dict(font=dict(size=10)))
    fig.update_xaxes(range=[0, 100])
    return style(fig, h)


@st.cache_resource(show_spinner=False)
def warm_models():
    from diverge import models
    models.embed(["warm"])
    models.nli([("a b c d", "e f g h")])
    models.perplexity("warm up the language model.")
    return True


def render_breakdown(ev, baselines):
    a, b = st.columns([1, 1.25])
    with a:
        st.plotly_chart(gauge(ev["score"]), theme=None, config=PLOT_CFG)
        c1, c2, c3 = st.columns(3)
        card(c1, "Originality", f"{ev['originality']:.0f}", "uncredited, 0-100")
        card(c2, "Quality", f"{ev['quality']:.0f}", "reasoning + evidence")
        card(c3, "Quality gate", f"{ev['gate']:.0%}", "originality credited", ev["gate"] >= 0.999)
    with b:
        st.plotly_chart(radar(ev), theme=None, config=PLOT_CFG)
    st.plotly_chart(contributions_bar(ev), theme=None, config=PLOT_CFG)
    st.markdown("##### Why this score")
    for line in ev["explanation"]:
        st.markdown(f"- {line}")


# ---------------------------------------------------------------- header
recs = store.load_all()
df_all = store.table(recs)
st.markdown(f"<div class='dv-nav'><div><span class='dv-logo'>DI<b>VERGE</b></span>"
            f"<span class='dv-pill'>Human Intelligence Exchange</span><span class='dv-pill lime'>{VERSION}</span></div>"
            f"<div class='dv-tag'>{len(recs)} submissions · {df_all.user.nunique() if len(df_all) else 0} contributors</div></div>",
            unsafe_allow_html=True)

T_ARENA, T_BREAK, T_BOARD, T_DATA, T_EXP, T_CLIMATE, T_METHOD = st.tabs(
    ["◇ Arena", "Consensus Breaker", "Leaderboard", "Training Data", "Compute Experiment", "Climate Impact", "Method"])

# ================================================================ ARENA
with T_ARENA:
    st.markdown("<div class='dv-hero'><h2>Out-think the consensus.</h2><p>Read what the AI says. Then answer the way it "
                "wouldn't, and show your reasoning. Originality only counts if the reasoning holds up.</p></div>",
                unsafe_allow_html=True)
    left, right = st.columns([1.05, 1])
    with left:
        user = st.text_input("Handle", value=st.session_state.get("user", ""), placeholder="your-handle", key="user_in")
        qopt = {f"{q['domain']} · {q['text']}": q["id"] for q in QUESTIONS}
        qlabel = st.selectbox("Challenge", list(qopt), key="arena_q")
        q = BY_ID[qopt[qlabel]]
        st.markdown(f"<span class='dv-pill'>{q['domain']}</span><span class='dv-pill warn'>{q['difficulty']}</span>"
                    f"<div class='dv-q'>{q['text']}</div>", unsafe_allow_html=True)
        answer = st.text_area("Your answer", height=130, key="arena_answer",
                              placeholder="State your claim. What does the conventional answer miss?")
        reasoning = st.text_area("Your reasoning & evidence", height=150, key="arena_reasoning",
                                 placeholder="Why is it true? Mechanisms, examples, numbers, sources, and a prediction that would prove you wrong.")
        go_btn = st.button("Submit & score", type="primary", disabled=not answer.strip())
    with right:
        with st.spinner("Loading AI baselines…"):
            base = get_baselines(q["text"])
        st.markdown(f"##### AI baseline answers <span class='dv-pill'>{base['meta']['model'].split('/')[-1]} · n={base['meta']['n']}</span>",
                    unsafe_allow_html=True)
        for i, btxt in enumerate(base["baselines"], 1):
            st.markdown(f"<div class='dv-base'><b>Baseline {i}</b><br>{btxt[:520]}{'…' if len(btxt) > 520 else ''}</div>",
                        unsafe_allow_html=True)
        st.caption("Generated locally by a small open model prompted for the conventional explanation; cached for reproducibility.")

    if go_btn:
        st.session_state["user"] = user.strip() or "anonymous"
        with st.spinner("Scoring: embeddings, NLI assumption checks, LM fluency…"):
            warm_models()
            ev = score_submission(q["text"], answer, reasoning, base["baselines"], store.prior_texts(q["id"]))
        rec = store.add(dict(user=st.session_state["user"], question_id=q["id"], domain=q["domain"], question=q["text"],
                             answer=answer.strip(), reasoning=reasoning.strip(), baselines=base["baselines"],
                             baseline_meta=base["meta"], evaluation=ev, demo_seed=False))
        st.session_state["last"] = rec
    last = st.session_state.get("last")
    if last and last["question_id"] == q["id"]:
        st.divider()
        ev = last["evaluation"]
        same_q = store.table()
        same_q = same_q[same_q.question_id == q["id"]]
        rank = int((same_q.score > ev["score"]).sum()) + 1
        st.markdown(f"<div class='dv-insight'>Divergence Score <b>{ev['score']:.1f}</b> · rank <b>#{rank}</b> of {len(same_q)} "
                    f"on this challenge · {'training-grade ✓' if ev['score'] >= store.TRAINING_THRESHOLD else 'below training-grade threshold'} "
                    f"(≥{store.TRAINING_THRESHOLD:.0f}).</div>", unsafe_allow_html=True)
        render_breakdown(ev, last["baselines"])

# ================================================================ CONSENSUS BREAKER
with T_BREAK:
    st.markdown("<div class='dv-hero'><h2>Consensus Breaker</h2><p>Sample several AI answers, extract the assumptions "
                "they share, then test whether a human argument actually contradicts them, and whether it still holds together.</p></div>",
                unsafe_allow_html=True)
    cb_mode = st.radio("Question", ["From the challenge bank", "Custom question"], horizontal=True, key="cb_mode")
    if cb_mode == "From the challenge bank":
        cq = BY_ID[st.selectbox("Challenge", [q["id"] for q in QUESTIONS], format_func=lambda i: BY_ID[i]["text"], key="cb_q")]["text"]
    else:
        cq = st.text_input("Your question", key="cb_custom", placeholder="Ask something where the obvious answer might be wrong")
    if cq:
        cb = get_baselines(cq, generate_if_missing=False)
        if cb is None:
            if st.button("Generate 3 AI baselines (~20 s, local model)", key="cb_gen"):
                with st.spinner("Sampling baseline answers…"):
                    cb = get_baselines(cq)
        if cb is not None:
            assumptions = consensus_assumptions(cb["baselines"])
            a, b = st.columns([1, 1])
            with a:
                st.markdown("##### ① AI baselines")
                for i, btxt in enumerate(cb["baselines"], 1):
                    with st.expander(f"Baseline {i}", expanded=False):
                        st.write(btxt)
                st.markdown("##### ② Shared assumptions")
                for asm in assumptions:
                    st.markdown(f"<div class='dv-assume'><div class='t'>{asm['text']}</div><div class='m'>shared by "
                                f"{asm['support']}/{len(cb['baselines'])} baselines · centrality {asm['centrality']:.2f}</div></div>",
                                unsafe_allow_html=True)
                st.caption("Sentences whose meaning recurs (cosine ≥ 0.6) across independent samples.")
            with b:
                st.markdown("##### ③ Your alternative reasoning")
                cb_ans = st.text_area("Argument", height=200, key="cb_answer",
                                      placeholder="Challenge one of the shared assumptions directly, and explain why.")
                run_cb = st.button("Break the consensus", type="primary", disabled=not cb_ans.strip(), key="cb_run")
            if run_cb:
                with st.spinner("Running NLI against each assumption…"):
                    warm_models()
                    ev = score_submission(cq, cb_ans, "", cb["baselines"], [])
                st.session_state["cb_last"] = (cq, ev)
            if st.session_state.get("cb_last") and st.session_state["cb_last"][0] == cq:
                ev = st.session_state["cb_last"][1]
                st.divider()
                st.markdown("##### ④ Assumption challenge")
                for row in ev["assumptions"]:
                    c, e = row["contradiction"], row["entailment"]
                    status = ("<span class='dv-ok'>CHALLENGED</span>" if c >= 0.5 else
                              "<span class='dv-no'>ACCEPTED</span>" if e >= 0.5 else "<span class='dv-mid'>NOT ADDRESSED</span>")
                    st.markdown(f"<div class='dv-assume'><div class='t'>{status} · {row['assumption']}</div>"
                                f"<div class='m'>contradiction {c:.2f} · entailment {e:.2f} · strongest counter-claim: "
                                f"“{row['strongest_challenge'][:180]}”</div></div>", unsafe_allow_html=True)
                rq = ev["details"]["reasoning"]
                k1, k2, k3, k4 = st.columns(4)
                card(k1, "Assumption challenge", f"{ev['details']['assumption_challenge']:.2f}", "support-weighted NLI contradiction",
                     ev["details"]["assumption_challenge"] >= 0.5)
                card(k2, "⑤ Coherence", f"{ev['components']['reasoning_quality']:.2f}",
                     f"self-consistency {rq['self_consistency']:.2f} · ppl {rq['perplexity']}")
                defensible = ev["gate"] >= 0.999 and ev["components"]["evidence_quality"] >= 0.3
                card(k3, "Defensible?", "Yes" if defensible else ("Partly" if ev["gate"] > 0 else "No"),
                     f"gate {ev['gate']:.0%} · evidence {ev['components']['evidence_quality']:.2f}", defensible)
                card(k4, "Divergence Score", f"{ev['score']:.1f}", "same scorer as the Arena", True)
                st.markdown("##### ⑥ New information contributed")
                new = [c for c in ev["claims"] if c["nearest_similarity"] < CLAIM_NEW_SIM]
                if new:
                    for c in new:
                        st.markdown(f"- {c['claim']} <span class='dv-muted'>(closest baseline sentence cos {c['nearest_similarity']:.2f})</span>",
                                    unsafe_allow_html=True)
                else:
                    st.markdown("No claim is far enough from the baselines to count as new information.")
                st.caption(f"Information gain {ev['components']['information_gain']:.2f}: share of the argument's embedding not "
                           "spanned by the baseline answers.")

# ================================================================ LEADERBOARD
with T_BOARD:
    df = store.table()
    if df.empty:
        st.info("No submissions yet. Be the first in the Arena.")
    else:
        show_demo = st.toggle("Include demo seed entries", value=True)
        d = df if show_demo else df[~df.demo_seed]
        lb = store.leaderboard(d)
        c1, c2, c3, c4 = st.columns(4)
        card(c1, "Top score", f"{d.score.max():.1f}", lb.iloc[0].user if len(lb) else "", True)
        card(c2, "Submissions", f"{len(d)}", f"{d.question_id.nunique()} challenges")
        card(c3, "Training-grade", f"{int((d.score >= store.TRAINING_THRESHOLD).sum())}", f"score ≥ {store.TRAINING_THRESHOLD:.0f}")
        card(c4, "Gated out", f"{int((d.gate < 0.5).sum())}", "low-quality divergence blocked")
        st.write("")
        a, b = st.columns([1, 1.2])
        with a:
            st.markdown("##### Divergence Leaderboard")
            st.dataframe(lb.assign(rank=range(1, len(lb) + 1))[["rank", "user", "best", "mean", "submissions", "training_grade", "demo"]].round(1),
                         width="stretch", hide_index=True, height=420,
                         column_config={"best": st.column_config.ProgressColumn("Best", min_value=0, max_value=100, format="%.1f"),
                                        "demo": "Demo seed"})
        with b:
            fig = go.Figure(go.Scatter(
                x=d.originality, y=d.quality, mode="markers", text=d.user,
                marker=dict(size=10 + d.score / 4, color=d.score, colorscale=[[0, "#2A1F48"], [0.5, PURPLE], [1, LIME]],
                            cmin=0, cmax=100, showscale=True, colorbar=dict(title="Score", thickness=10),
                            line=dict(color="#07060C", width=1)),
                customdata=np.stack([d.score, d.question_id], axis=1),
                hovertemplate="%{text} · %{customdata[1]}<br>originality %{x:.0f} · quality %{y:.0f}<br>score %{customdata[0]:.1f}<extra></extra>"))
            fig.add_shape(type="rect", x0=50, x1=102, y0=50, y1=102, line=dict(color=LIME, dash="dot"), fillcolor="rgba(198,255,61,0.04)")
            fig.add_annotation(x=100, y=100, text="divergent & sound", showarrow=False, xanchor="right", font=dict(color=LIME))
            fig.add_annotation(x=100, y=4, text="different but unsupported → gated", showarrow=False, xanchor="right",
                               font=dict(color="#FF6B8B", size=10))
            style(fig, 420, "Originality vs. quality: scored separately")
            fig.update_xaxes(title="Originality (0-100)", range=[-2, 104])
            fig.update_yaxes(title="Quality (0-100)", range=[-2, 104])
            st.plotly_chart(fig, theme=None, config=PLOT_CFG)
        st.markdown("##### All submissions")
        st.dataframe(d.sort_values("score", ascending=False)[["user", "question_id", "score", "originality", "quality", "gate", "answer"]].round(2),
                     width="stretch", hide_index=True)

# ================================================================ TRAINING DATA
with T_DATA:
    df = store.table()
    if df.empty:
        st.info("No submissions to export yet.")
    else:
        st.markdown("<div class='dv-hero'><h2>Training Data Engine</h2><p>Select training-grade human reasoning and export "
                    "it as JSONL with baselines, component scores and full evaluation metadata.</p></div>", unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        thr = f1.slider("Minimum Divergence Score", 0, 100, int(store.TRAINING_THRESHOLD))
        min_gate = f2.slider("Minimum quality gate", 0.0, 1.0, 0.99, 0.01)
        fmt = f3.selectbox("Format", ["DIVERGE record (full)", "chat (fine-tuning)", "preference pair (DPO)"])
        incl_demo = f4.toggle("Include demo seeds", value=True)
        sel = df[(df.score >= thr) & (df.gate >= min_gate) & (incl_demo | ~df.demo_seed)].copy()
        sel.insert(0, "include", True)
        edited = st.data_editor(sel[["include", "id", "user", "question_id", "score", "originality", "quality", "answer"]].round(1),
                                hide_index=True, width="stretch", disabled=["id", "user", "question_id", "score", "originality", "quality", "answer"],
                                key="export_editor")
        ids = edited[edited.include].id.tolist()
        payload = store.export_jsonl(ids, fmt) if ids else ""
        c1, c2 = st.columns([1, 3])
        c1.download_button(f"Export {len(ids)} records · JSONL", payload, file_name=f"diverge_{fmt.split()[0].lower()}.jsonl",
                           mime="application/jsonl", disabled=not ids)
        c2.caption(f"{len(ids)} of {len(df)} submissions selected. Demo seeds are flagged in metadata (demo_seed=true).")
        if ids:
            with st.expander("Preview first record"):
                st.json(json.loads(payload.splitlines()[0]))

# ================================================================ COMPUTE EXPERIMENT
res = X.load_results()
summ = X.summarize(res) if res else None
with T_EXP:
    st.markdown("<div class='dv-hero'><h2>Compute-efficiency experiment</h2><p>Same model, same optimizer, same epochs. "
                "Only the <i>selection</i> of training examples changes: consensus-typical vs. divergent vs. random.</p></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='dv-measured'>● MEASURED · small-scale proxy: frozen MiniLM embeddings + 2-layer MLP trained from "
                "scratch on CommonsenseQA (human-written, gold-labeled) subsets; evaluated on the 1,221-question held-out "
                "validation split (chance 20%). Not a fine-tune of an LLM, and not trained on DIVERGE submissions yet.</div>",
                unsafe_allow_html=True)
    if res is None:
        st.warning("No results yet. Run `.venv/bin/python -m diverge.experiment` (~2-3 min on a laptop CPU).")
    else:
        sig = X.significance(res)
        ctt = pd.DataFrame([r for t in res["config"]["target_accuracies"]
                            for r in X.compute_to_target(summ, t).to_dict("records")])
        small = sig[sig.budget <= 400]
        k1, k2, k3, k4 = st.columns(4)
        card(k1, "Divergent − conventional", f"{small['diff'].mean() * 100:+.1f} pts", "mean accuracy, N ≤ 400", True)
        big = sig[sig.budget == max(sig.budget)].iloc[0]
        card(k2, f"At N = {int(big.budget):,}", f"{big['diff'] * 100:+.1f} pts", "advantage disappears at scale")
        g = summ.set_index(["arm", "budget"]).acc_mean
        card(k3, "Divergent − random", f"{(g['divergent'] - g['random']).mean() * 100:+.1f} pts", "mean over all budgets")
        card(k4, "Runs", f"{len(res['rows'])}", f"{len(res['config']['seeds'])} seeds × {len(res['config']['budgets'])} budgets × 3 arms")

        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            for arm in res["config"]["arms"]:
                s = summ[summ.arm == arm]
                fig.add_trace(go.Scatter(x=s.flops, y=s.acc_mean * 100, mode="lines+markers", name=arm,
                                         line=dict(color=ARM_COLORS[arm], width=2.5),
                                         error_y=dict(type="data", array=s.acc_std * 100, color=ARM_COLORS[arm], thickness=1),
                                         customdata=s.budget, hovertemplate=arm + " N=%{customdata}<br>%{y:.1f}% @ %{x:.2e} FLOPs<extra></extra>"))
            fig.add_hline(y=20, line=dict(color="#3A3355", dash="dot"), annotation_text="chance")
            style(fig, 380, "Held-out accuracy vs. training compute (mean ± sd, 5 seeds)")
            fig.update_xaxes(type="log", title="training FLOPs (analytic: 6 × params × pairs)")
            fig.update_yaxes(title="accuracy %")
            st.plotly_chart(fig, theme=None, config=PLOT_CFG)
        with b:
            fig = go.Figure()
            for arm in res["config"]["arms"]:
                s = summ[summ.arm == arm]
                fig.add_trace(go.Scatter(x=s.budget, y=s.redundancy, mode="lines+markers", name=arm, line=dict(color=ARM_COLORS[arm], width=2.5)))
            style(fig, 380, "Training-set redundancy (mean pairwise cosine)")
            fig.update_xaxes(type="log", title="examples (N)")
            fig.update_yaxes(title="lower = more divergent")
            st.plotly_chart(fig, theme=None, config=PLOT_CFG)

        st.markdown("##### Compute to reach a target accuracy (smallest tested budget whose mean reaches it)")
        piv = ctt.pivot(index="target", columns="arm", values="flops")
        rows = []
        for t, r in piv.iterrows():
            ratio = r["conventional"] / r["divergent"] if pd.notna(r.get("conventional")) and pd.notna(r.get("divergent")) else np.nan
            rows.append({"target accuracy": f"{t:.0%}", **{f"{arm} FLOPs": (f"{r[arm]:.2e}" if pd.notna(r[arm]) else "not reached") for arm in piv.columns},
                         "conventional ÷ divergent": f"{ratio:.1f}×" if pd.notna(ratio) else "n/a"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("##### Paired test, divergent vs. conventional (sign-flip permutation over seeds)")
        st.dataframe(sig.assign(diff=lambda x: (x["diff"] * 100).round(2)).rename(columns={"diff": "Δ accuracy (pts)"}).round(4),
                     hide_index=True, width="stretch")
        st.markdown(f"<div class='dv-insight'><b>What the data supports:</b> selecting divergent (non-redundant) examples beat "
                    f"selecting consensus-typical examples at small budgets, and needed fewer FLOPs to reach the lower accuracy "
                    f"targets. <b>What it does not support:</b> divergent selection did not beat random selection, and the gap "
                    f"vanished at N = {int(big.budget):,}. With 5 seeds the smallest attainable p-value is 0.0625, so no "
                    f"difference here is significant at 0.05. Treat this as a reproducible pilot, not evidence of energy savings.</div>",
                    unsafe_allow_html=True)
        with st.expander("Reproduce / environment"):
            st.code(".venv/bin/python -m diverge.experiment", language="bash")
            st.json(dict(config=res["config"], environment=res["environment"], completed_at=res["completed_at"],
                         n_train_pool=res["n_train_pool"], n_heldout=res["n_heldout"],
                         embedding_seconds_one_time=round(res["embedding_seconds_one_time"], 1)))
        if st.button("Re-run experiment now (~2-3 min)", key="rerun_exp"):
            bar = st.progress(0.0)
            X.run(progress=lambda f, m: bar.progress(f, text=m))
            st.rerun()

# ================================================================ CLIMATE IMPACT
with T_CLIMATE:
    st.markdown("<div class='dv-hero'><h2>From human reasoning to energy</h2><p>The causal chain DIVERGE is testing. "
                "Green links are measured in our pilot; amber links are hypotheses.</p></div>", unsafe_allow_html=True)
    nodes = ["Human reasoning", "Higher-information datasets", "Model improvement", "Compute efficiency", "Potential energy savings"]
    status = ["measured", "measured", "measured", "hypothesis"]
    fig = go.Figure()
    for i, n in enumerate(nodes):
        fig.add_trace(go.Scatter(x=[i], y=[0], mode="markers+text", text=[n], textposition="bottom center",
                                 marker=dict(size=46, color=[PURPLE, VIOLET, "#B6E35A", LIME, "#FFC857"][i],
                                             line=dict(color="#07060C", width=3)),
                                 textfont=dict(family="Space Grotesk", size=13, color=INK), showlegend=False, hoverinfo="skip"))
    if res:
        redu = summ[summ.budget == 400].set_index("arm").redundancy
        dlt = sig[sig.budget <= 400]["diff"].mean() * 100
        labels = ["DIVERGE Score (live)", f"redundancy {redu['conventional']:.2f} → {redu['divergent']:.2f}",
                  f"{dlt:+.1f} pts vs conventional (N≤400)", "not yet: kWh unmeasured at scale"]
    else:
        labels = ["DIVERGE Score", "run experiment", "run experiment", "hypothesis"]
    for i in range(4):
        colr = LIME if status[i] == "measured" else "#FFC857"
        fig.add_annotation(x=i + 0.92, y=0, ax=i + 0.08, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=2.5, arrowcolor=colr)
        fig.add_annotation(x=i + 0.5, y=0.32, text=labels[i], showarrow=False, font=dict(size=11, color=colr))
    fig.update_xaxes(visible=False, range=[-0.6, 4.6])
    fig.update_yaxes(visible=False, range=[-0.8, 0.7])
    st.plotly_chart(style(fig, 260), theme=None, config=PLOT_CFG)
    st.caption("“Model improvement” link: measured as a selection effect between consensus-typical and divergent subsets in a "
               "small proxy model; it did not beat random selection.")

    st.markdown("<div class='dv-measured'>● MEASURED (pilot, this laptop)</div>", unsafe_allow_html=True)
    if res:
        m = st.columns(4)
        tot_s = sum(r["train_seconds"] for r in res["rows"])
        card(m[0], "Total training wall-clock", f"{tot_s:.1f} s", f"{len(res['rows'])} runs, CPU")
        card(m[1], "Total CPU time", f"{sum(r['train_cpu_seconds'] for r in res['rows']):.1f} s", "process time")
        card(m[2], "Total training FLOPs", f"{sum(r['train_flops'] for r in res['rows']):.2e}", "analytic")
        piv30 = X.compute_to_target(summ, 0.30).set_index("arm").flops
        r30 = piv30["conventional"] / piv30["divergent"] if pd.notna(piv30.get("conventional")) and pd.notna(piv30.get("divergent")) else np.nan
        card(m[3], "FLOPs to 30% acc.", f"{r30:.1f}× fewer" if pd.notna(r30) else "n/a", "divergent vs conventional (tested budgets)", True)

    st.markdown("<div class='dv-hypo'>▲ HYPOTHETICAL SCENARIO: every number below is a user-set assumption. "
                "Our pilot does not show that divergent data reduces electricity use in real model training.</div>",
                unsafe_allow_html=True)
    a, b = st.columns([1, 1.2])
    with a:
        run_mwh = st.number_input("Energy of one training / fine-tuning run (MWh)", 0.1, 100000.0, 1000.0, 50.0,
                                  help="Assumption. Set to your workload; public estimates for frontier runs vary by orders of magnitude.")
        runs_per_year = st.number_input("Runs per year", 1, 10000, 10)
        data_reduction = st.slider("Assumed compute reduction from better data", 0, 80, 10, 1, format="%d%%",
                                   help="Assumption. Our pilot's lowest-target ratio is shown alongside, but it does not transfer to LLM training.") / 100
        overhead = st.slider("Data-curation overhead (% of saved energy spent on scoring)", 0, 100, 5, 1) / 100
        ci = st.number_input("Grid carbon intensity (kg CO₂ / MWh)", 0.0, 1000.0, 350.0, 10.0, help="Assumption; varies by region and hour.")
        pue = st.number_input("Data-center PUE", 1.0, 2.0, 1.2, 0.05)
    with b:
        gross = run_mwh * runs_per_year * pue * data_reduction
        net = gross * (1 - overhead)
        tco2 = net * ci / 1000
        c1, c2 = st.columns(2)
        card(c1, "Hypothetical energy avoided", f"{net:,.0f} MWh/yr", "scenario, not measured", True)
        card(c2, "Hypothetical emissions avoided", f"{tco2:,.0f} t CO₂/yr", f"at {ci:.0f} kg/MWh")
        xs = np.arange(0, 81, 2) / 100
        fig = go.Figure(go.Scatter(x=xs * 100, y=run_mwh * runs_per_year * pue * xs * (1 - overhead), mode="lines",
                                   line=dict(color="#FFC857", width=2.5), fill="tozeroy", fillcolor="rgba(255,200,87,0.08)"))
        fig.add_vline(x=data_reduction * 100, line=dict(color=INK, dash="dot"), annotation_text="your assumption")
        style(fig, 300, "Sensitivity: energy avoided vs. assumed compute reduction")
        fig.update_xaxes(title="assumed compute reduction %")
        fig.update_yaxes(title="MWh / yr")
        st.plotly_chart(fig, theme=None, config=PLOT_CFG)
        st.caption("energy avoided = run MWh × runs/yr × PUE × reduction × (1 − overhead). "
                   "CO₂ = energy × carbon intensity.")

# ================================================================ METHOD
with T_METHOD:
    st.markdown("#### DIVERGE Score")
    st.latex(r"S = 100\,[\,g\,(0.30\,D + 0.25\,N + 0.10\,I) + 0.25\,R + 0.10\,E\,]")
    st.markdown(f"""
Weights are **initial experimental assumptions, not validated measurements**. *g* is a quality gate,
`clip((R − 0.25)/0.35, 0, 1)`, so originality only counts when the reasoning is coherent.

| Component | How it is computed |
|---|---|
| **D** Semantic divergence (30%) | 1 − max cosine between the answer's sentence embedding and each AI baseline, calibrated so 0.85 → 0 and 0.25 → 1 |
| **N** Conceptual novelty (25%) | 0.6 × share of the answer's claims with no close counterpart in baselines/prior submissions + 0.4 × NLI contradiction of the consensus assumptions |
| **R** Reasoning quality (25%) | relevance to the question, fluency (perplexity under the local LM), self-consistency (NLI between the answer's own sentences), causal structure, depth |
| **E** Evidence quality (10%) | transparent pattern checks: quantities, named examples, mechanisms, sources, testable predictions. Detects presence, **not truth** |
| **I** Marginal information gain (10%) | residual of the answer embedding after projection onto the span of all existing answers (1 − kᵀK⁻¹k) |

**Models (all local, no data leaves the machine):** all-MiniLM-L6-v2 embeddings · nli-deberta-v3-xsmall cross-encoder ·
Qwen2.5-0.5B-Instruct for baselines and perplexity.

**Known limitations:** no fact-checking, so a fluent, well-structured false claim can still score well on quality.
Evidence detection is lexical. The NLI model is small and can misread long sentences. Calibration anchors were set by inspection,
not fitted to human ratings. The next step is collecting expert pairwise judgments to fit the weights.
""")
    st.markdown("#### Repository")
    st.markdown("`diverge/` holds the platform (scoring, baselines, store, experiment). `engine/divergence.py` holds the embedding "
                "utilities reused from the earlier project. The earlier UNBOUND grid app is preserved at `unbound/unbound_app.py`.")
