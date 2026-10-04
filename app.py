"""DecisionOS - AI Decision & Negotiation Engine (Streamlit UI)."""
import config  # noqa: F401  (must be first: sets env vars before crewai loads)
import copy
import pandas as pd
import streamlit as st

from config import get_api_key
from decision_engine import (CRITERIA, LABELS, DEFAULT_WEIGHTS, RISK_SCORES,
                             score_options, validate_options, weight_sensitivity)

st.set_page_config(page_title="DecisionOS", page_icon="🧭", layout="wide")

DEMO = pd.DataFrame([
    {"name": "Supplier Alpha", "cost": 100000, "delivery": 30, "quality": 95, "risk": "Medium", "strategic": None, "source": "Procurement quote"},
    {"name": "Supplier Beta", "cost": 85000, "delivery": 45, "quality": 91, "risk": "Low", "strategic": None, "source": "Procurement quote"},
    {"name": "Supplier Gamma", "cost": 120000, "delivery": 20, "quality": 98, "risk": "Medium", "strategic": None, "source": "Procurement quote"},
])

STEP_TITLES = {"finance": "💰 Finance Analyst", "risk": "⚠️ Risk Analyst",
               "operations": "⚙️ Operations Analyst", "negotiation": "🤝 Negotiation Strategist",
               "decision": "🧭 Decision Synthesizer"}

ss = st.session_state
ss.setdefault("options_df", DEMO.copy())
ss.setdefault("title", "Select a technology implementation supplier")
ss.setdefault("objective", "Implementation should ideally be completed within 30 days.")
ss.setdefault("analysis", None)       # dict with findings + frozen inputs
ss.setdefault("scenario_text", None)


def df_to_options(df: pd.DataFrame) -> list[dict]:
    opts = []
    for rec in df.to_dict("records"):
        if not str(rec.get("name") or "").strip() and pd.isna(rec.get("cost")):
            continue  # skip fully empty rows
        rec["name"] = str(rec.get("name") or "").strip()
        rec["source"] = "" if pd.isna(rec.get("source")) else str(rec.get("source"))
        if pd.isna(rec.get("strategic")):
            rec["strategic"] = None
        opts.append(rec)
    return opts


COLUMN_CFG = {
    "name": st.column_config.TextColumn("Option", required=True),
    "cost": st.column_config.NumberColumn("Cost", min_value=1, format="%d", required=True),
    "delivery": st.column_config.NumberColumn("Delivery (days)", min_value=1, required=True),
    "quality": st.column_config.NumberColumn("Quality (0-100)", min_value=0, max_value=100, required=True),
    "risk": st.column_config.SelectboxColumn("Risk", options=list(RISK_SCORES), required=True),
    "strategic": st.column_config.NumberColumn("Strategic value (0-100, optional)", min_value=0, max_value=100),
    "source": st.column_config.TextColumn("Data source / owner"),
}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚖️ Criteria weights")
    st.caption("Any numbers work - they are converted to percentages.")
    weights = {c: st.slider(LABELS[c], 0, 100, DEFAULT_WEIGHTS[c], key=f"w_{c}") for c in CRITERIA}
    total = sum(weights.values()) or 1
    st.caption("Effective: " + " · ".join(f"{LABELS[c]} {weights[c] / total * 100:.0f}%" for c in CRITERIA))
    st.divider()
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Groq API key", type="password",
                                help="Or add GROQ_API_KEY to Streamlit secrets.")
    st.caption("Model: openai/gpt-oss-120b on Groq")

st.title("🧭 DecisionOS")
st.caption("AI Decision & Negotiation Engine — a virtual team of agents analyzes your trade-offs; "
           "Python (not the AI) calculates the scores.")

tab_setup, tab_results, tab_whatif, tab_summary = st.tabs(
    ["1️⃣ Decision setup", "2️⃣ Results", "3️⃣ What-If simulator", "4️⃣ Executive summary"])

# ---------------- Tab 1: setup ----------------
with tab_setup:
    ss.title = st.text_input("Decision title", ss.title)
    ss.objective = st.text_area("Business context / requirements (optional)", ss.objective, height=70)
    c1, c2 = st.columns([1, 5])
    if c1.button("Load demo data"):
        ss.options_df = DEMO.copy()
        st.rerun()
    st.markdown("**Options (2-4 rows).** Quality, risk and strategic value are *business inputs* "
                "from your QA scores, risk register or management — not facts invented by AI.")
    edited = st.data_editor(ss.options_df, column_config=COLUMN_CFG, num_rows="dynamic",
                            hide_index=True, width="stretch", key="editor")
    options = df_to_options(edited)
    problems = validate_options(options)
    for p in problems:
        st.warning(p)

    run = st.button("🚀 Run analysis", type="primary", disabled=bool(problems) or not api_key)
    if not api_key:
        st.info("Add your Groq API key in the sidebar (or Streamlit secrets) to enable the agents.")

    if run:
        from crew import run_analysis  # imported lazily so the page loads fast
        ss.scenario_text = None
        status = st.status("Agents are working…", expanded=True)
        status.write("🧮 Python scoring engine calculated the weighted scores.")

        def on_step(key, role, text):
            status.write(f"✅ {STEP_TITLES.get(key, role)} finished")

        try:
            findings = run_analysis(ss.title, ss.objective, options, weights,
                                    api_key=api_key, on_step_done=on_step)
            ss.analysis = {"findings": findings, "options": copy.deepcopy(options),
                           "weights": dict(weights), "title": ss.title, "objective": ss.objective}
            status.update(label="Analysis complete — open the Results tab", state="complete", expanded=False)
        except Exception as e:  # show a friendly message instead of a stack trace
            status.update(label="Analysis failed", state="error")
            st.error(f"The agents could not finish: {e}")
            st.caption("Common causes: invalid API key, Groq rate limit (wait a minute and retry), or no network.")

# ---------------- Tab 2: results ----------------
with tab_results:
    a = ss.analysis
    if not a:
        st.info("Run the analysis first.")
    else:
        scores = score_options(a["options"], a["weights"])
        st.subheader("🏆 Transparent Python scores")
        winner = scores.index[0]
        st.success(f"Highest weighted score: **{winner}** ({scores['Final Score'].iloc[0]})")
        show = scores.rename(columns={c: LABELS[c] for c in CRITERIA})
        st.dataframe(show, width="stretch")
        st.bar_chart(scores["Final Score"])
        st.caption("Cost & delivery: best option = 100, others = 100 × best ÷ value. "
                   "Risk: Low 90 / Medium 70 / High 40 (demo convention). Empty strategic value = 50 (neutral).")
        st.subheader("🔍 How fragile is the ranking?")
        st.dataframe(weight_sensitivity(a["options"], a["weights"]), hide_index=True, width="stretch")
        st.subheader("🤖 Agent findings")
        for key in ["finance", "risk", "operations", "negotiation"]:
            with st.expander(STEP_TITLES[key], expanded=False):
                st.markdown(a["findings"].get(key, "_No output_"))

# ---------------- Tab 3: what-if ----------------
with tab_whatif:
    a = ss.analysis
    if not a:
        st.info("Run the analysis first, then change assumptions here.")
    else:
        st.markdown("Change weights or option values and see the **Python scores** update instantly.")
        wc = st.columns(len(CRITERIA))
        s_weights = {c: wc[i].slider(f"{LABELS[c]} weight", 0, 100, int(a["weights"][c]), key=f"s_{c}")
                     for i, c in enumerate(CRITERIA)}
        base_df = pd.DataFrame(a["options"])
        s_df = st.data_editor(base_df, column_config=COLUMN_CFG, hide_index=True,
                              width="stretch", key="scenario_editor",
                              disabled=["name", "source"])
        s_options = df_to_options(s_df)
        s_problems = validate_options(s_options)
        for p in s_problems:
            st.warning(p)
        if not s_problems:
            base_scores = score_options(a["options"], a["weights"])
            scen_scores = score_options(s_options, s_weights)
            cmp_df = pd.DataFrame({
                "Baseline": base_scores["Final Score"],
                "Scenario": scen_scores["Final Score"]}).fillna(0)
            cmp_df["Change"] = (cmp_df["Scenario"] - cmp_df["Baseline"]).round(1)
            m1, m2 = st.columns(2)
            m1.metric("Baseline winner", base_scores.index[0])
            m2.metric("Scenario winner", scen_scores.index[0],
                      delta="changed" if scen_scores.index[0] != base_scores.index[0] else "same",
                      delta_color="off")
            st.bar_chart(cmp_df[["Baseline", "Scenario"]])
            st.dataframe(cmp_df, width="stretch")
            if st.button("🧠 Explain this scenario with AI", disabled=not api_key):
                from crew import run_scenario_summary
                with st.spinner("Decision Synthesizer is analyzing the scenario…"):
                    try:
                        ss.scenario_text = run_scenario_summary(
                            a["title"], a["objective"], a["options"], a["weights"],
                            s_options, s_weights, a["findings"], api_key=api_key)
                    except Exception as e:
                        st.error(f"Could not generate explanation: {e}")
            if ss.scenario_text:
                st.markdown(ss.scenario_text)

# ---------------- Tab 4: summary ----------------
with tab_summary:
    a = ss.analysis
    if not a:
        st.info("Run the analysis first.")
    else:
        st.markdown(a["findings"].get("decision", "_No output_"))
        st.divider()
        st.caption("⚠️ DecisionOS is decision support, not an automated decision. Quality, risk and "
                   "strategic values are manually entered business inputs; in production they should "
                   "come from approved systems or documented assessments.")
        report = f"# {a['title']}\n\n" + a["findings"].get("decision", "") + "\n\n---\n\n" + "\n\n".join(
            f"## {STEP_TITLES[k]}\n{a['findings'].get(k, '')}" for k in ["finance", "risk", "operations", "negotiation"])
        st.download_button("⬇️ Download report (Markdown)", report, file_name="decisionos_report.md")
