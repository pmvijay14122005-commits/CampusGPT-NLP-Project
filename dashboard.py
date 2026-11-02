"""
dashboard.py - CampusGPT results dashboard (for the review).

    pip install plotly pandas
    streamlit run dashboard.py

It reads the files that evaluate.py and stress.py already wrote into results/:
    qa_summary.csv, qa_threshold_sweep.csv, quiz_results.csv, stress_results.csv
Anything missing is skipped, so it works even before every test has finished.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS = "results"

# Retrieval hit rate@4 is printed by evaluate.py but not saved to a file.
# Type your numbers from the terminal here (percent).
HIT_VECTOR = 94.4
HIT_HYBRID = 100.0

PALETTE = ["#6366F1", "#EC4899", "#F59E0B", "#10B981", "#06B6D4", "#8B5CF6"]

st.set_page_config(page_title="CampusGPT Results", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.hero {background: linear-gradient(120deg,#4F46E5 0%,#9333EA 50%,#EC4899 100%);
       padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.hero p {margin: 6px 0 0 0; opacity: .92; font-size: 1.02rem;}
.card {border-radius: 16px; padding: 16px 18px; color: white; text-align: left;
       box-shadow: 0 6px 18px rgba(0,0,0,.12);}
.card .v {font-size: 2rem; font-weight: 700; line-height: 1.1;}
.card .l {font-size: .85rem; opacity: .92; margin-top: 4px;}
.c1 {background: linear-gradient(135deg,#6366F1,#8B5CF6);}
.c2 {background: linear-gradient(135deg,#10B981,#06B6D4);}
.c3 {background: linear-gradient(135deg,#F59E0B,#EF4444);}
.c4 {background: linear-gradient(135deg,#EC4899,#8B5CF6);}
.note {background:#EEF2FF; border-left:5px solid #6366F1; padding:10px 14px; border-radius:8px; color:#312E81;}
</style>
""",
    unsafe_allow_html=True,
)


def load(name):
    path = os.path.join(RESULTS, name)
    if os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return None
    return None


def card(value, label, css):
    st.markdown(f'<div class="card {css}"><div class="v">{value}</div><div class="l">{label}</div></div>',
                unsafe_allow_html=True)


def short(name):
    name = str(name)
    if name.startswith("LLM alone"):
        return "LLM alone"
    if "vector" in name:
        return "RAG vector"
    if "hybrid search" in name and "Verified" not in name:
        return "RAG hybrid"
    if "default" in name:
        return "Verified (default)"
    return "Verified (tuned)"


qa = load("qa_summary.csv")
sweep = load("qa_threshold_sweep.csv")
quiz = load("quiz_results.csv")
stress = load("stress_results.csv")

st.markdown(
    '<div class="hero"><h1>📊 CampusGPT: Evidence-Verified RAG</h1>'
    "<p>Grounded answers and quizzes from student study material: results of our evaluation</p></div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ KPI row
if qa is not None:
    qa = qa.copy()
    qa["label"] = qa["system"].map(short)
    pctcols = [c for c in qa.columns if c not in ("system", "n", "label")]
    qa[pctcols] = qa[pctcols] * 100
    best = qa.sort_values("accuracy", ascending=False).iloc[0]
    llm = qa[qa["label"] == "LLM alone"]
    ver = qa[qa["label"] == "Verified (tuned)"]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        card(f"{int(qa['n'].iloc[0])}", "test questions", "c1")
    with k2:
        card(f"{best['accuracy']:.1f}%", f"best accuracy ({best['label']})", "c2")
    with k3:
        if len(llm):
            card(f"{llm['accuracy'].iloc[0]:.1f}%", "LLM alone, no documents", "c3")
    with k4:
        if len(ver):
            card(f"{ver['correct_refusal_rate'].iloc[0]:.0f}%", "unanswerable questions refused (verified)", "c4")
    st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🧪 Answer quality", "🎚️ Thresholds", "🔎 Retrieval", "📝 Quiz check", "🛡️ Stress test"]
)

# ------------------------------------------------------------ Answer quality
with tab1:
    if qa is None:
        st.warning("results/qa_summary.csv not found. Run: python evaluate.py qa --questions eval_questions.csv")
    else:
        names = {
            "accuracy": "Accuracy",
            "selective_accuracy": "Selective accuracy",
            "coverage": "Coverage (questions answered)",
            "wrong_answer_rate": "Wrong answers",
            "false_refusal_rate": "False refusals",
            "correct_refusal_rate": "Correct refusals",
            "false_answer_rate": "False answers (answered an unanswerable question)",
        }
        chosen = st.multiselect("Metrics to compare", list(names), default=["accuracy", "correct_refusal_rate", "false_answer_rate"],
                                format_func=lambda k: names[k])
        systems = st.multiselect("Systems", list(qa["label"]), default=list(qa["label"]))
        view = qa[qa["label"].isin(systems)]
        if chosen and len(view):
            long = view.melt(id_vars="label", value_vars=chosen, var_name="metric", value_name="value")
            long["metric"] = long["metric"].map(names)
            fig = px.bar(long, x="label", y="value", color="metric", barmode="group", text="value",
                         color_discrete_sequence=PALETTE, labels={"label": "", "value": "%", "metric": ""})
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(height=470, yaxis_range=[0, 112], legend_orientation="h", legend_y=-0.18,
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="note"><b>How to read it.</b> Accuracy counts correct answers plus correct refusals. '
            "Verification matters most on the <i>false answer</i> bars: it refuses questions the notes cannot answer. "
            "The test set is small, so one question moves a bar by several points.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Show the full table"):
            st.dataframe(qa.drop(columns=["label"]).round(1), use_container_width=True)

# ---------------------------------------------------------------- Thresholds
with tab2:
    if sweep is None:
        st.warning("results/qa_threshold_sweep.csv not found.")
    else:
        sweep = sweep.copy()
        metric_names = {
            "accuracy": "Accuracy",
            "false_answer_rate": "False answer rate",
            "coverage": "Coverage",
            "selective_accuracy": "Selective accuracy",
            "false_refusal_rate": "False refusal rate",
        }
        m = st.selectbox("Metric", list(metric_names), format_func=lambda k: metric_names[k])
        grid = sweep.pivot(index="support_threshold", columns="retrieval_threshold", values=m) * 100
        fig = px.imshow(grid, text_auto=".0f", aspect="auto", origin="lower",
                        color_continuous_scale="Viridis",
                        labels={"x": "Minimum retrieval strength", "y": "Minimum evidence support", "color": "%"})
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each cell is one setting of the two sliders in the app. Pick the one that balances "
                   "answering more questions against refusing unsupported ones.")

# ----------------------------------------------------------------- Retrieval
with tab3:
    fig = go.Figure(go.Bar(x=["Vector search", "Hybrid (BM25 + vector)"], y=[HIT_VECTOR, HIT_HYBRID],
                           marker_color=["#6366F1", "#EC4899"], text=[f"{HIT_VECTOR:.1f}%", f"{HIT_HYBRID:.1f}%"],
                           textposition="outside"))
    fig.update_layout(height=420, yaxis_range=[0, 115], yaxis_title="Gold page found in top 4 (%)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Retrieval hit rate@4 on questions with a known source page. The gap is one question, so treat it as a trend.")

# ---------------------------------------------------------------------- Quiz
with tab4:
    if quiz is None:
        st.warning("results/quiz_results.csv not found. Run: python evaluate.py quiz --file YourFile.pdf")
    else:
        g = quiz.groupby("group")["flawed"].agg(["count", "sum"]).reset_index()
        g["flawed_rate"] = 100 * g["sum"] / g["count"]
        labels = {"baseline": "Without verification", "verified_kept": "Verified (kept)",
                  "verified_rejected": "Rejected by verifier"}
        g["group"] = g["group"].map(labels)
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(g, x="group", y="flawed_rate", color="group", text="flawed_rate",
                         color_discrete_sequence=PALETTE, labels={"group": "", "flawed_rate": "Flawed MCQs (%)"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, height=380, yaxis_range=[0, 100], plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(g.rename(columns={"count": "MCQs judged", "sum": "Flawed"}).round(1), use_container_width=True)
        st.caption("Flawed = wrong key, more than one correct option, or an explanation not in the source text "
                   "(judged by a second model).")

# -------------------------------------------------------------------- Stress
with tab5:
    if stress is None:
        st.warning("results/stress_results.csv not found. Run: python stress.py --questions eval_questions.csv --n 25")
    else:
        part = st.radio("Part", ["answer", "mcq"], horizontal=True,
                        format_func=lambda p: "Answers" if p == "answer" else "MCQs")
        d = stress[stress["part"] == part]
        if d.empty:
            st.info("No rows for this part yet.")
        else:
            rate = d.groupby("type")["flag_any"].agg(["count", "mean"]).reset_index()
            rate["pct"] = rate["mean"] * 100
            rate["kind"] = rate["type"].map(lambda t: "False alarm (correct item flagged)" if t == "clean" else "Detected (planted error flagged)")
            fig = px.bar(rate, x="type", y="pct", color="kind", text="pct",
                         color_discrete_map={"False alarm (correct item flagged)": "#F59E0B",
                                             "Detected (planted error flagged)": "#10B981"},
                         labels={"type": "", "pct": "Flagged (%)", "kind": ""})
            fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig.update_layout(height=420, yaxis_range=[0, 115], legend_orientation="h", legend_y=-0.2,
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green bars: how many deliberately planted errors the verifier caught (higher is better). "
                       "Orange bar: correct items it wrongly flagged (lower is better).")
            with st.expander("Browse the planted errors"):
                kinds = ["all"] + sorted(d["type"].unique())
                pick = st.selectbox("Type", kinds)
                shown = d if pick == "all" else d[d["type"] == pick]
                st.dataframe(shown[["type", "question", "text", "flag_any"]], use_container_width=True)