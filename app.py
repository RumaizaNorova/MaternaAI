import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import label_binarize

from data import load_dataset, get_binary_df, FEATURE_COLS, FEATURE_LABELS, RISK_LABEL, RISK_COLOR
from model import train_models, predict_risk, get_feature_importance
from fairness_engine import run_audit, verdict
from granite import explain_risk, governance_policy, is_live

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaternaAI | IBM watsonx",
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0b0f19; }
.ibm-header {
    background: linear-gradient(135deg, #0530ad 0%, #0f62fe 60%, #6929c4 100%);
    padding:24px 32px; border-radius:14px; margin-bottom:18px;
}
.risk-card {
    border-radius:14px; padding:22px; text-align:center;
    border:2px solid; margin:4px;
}
.stat-pill {
    display:inline-block; border-radius:20px; padding:3px 14px;
    font-size:.82em; font-weight:700; margin:2px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ibm-header">
  <h1 style="color:white;margin:0;font-size:2rem">🤱 MaternaAI</h1>
  <p style="color:#a6c8ff;margin:6px 0 0">
    AI-Powered Maternal Health Risk Stratification for Underserved Communities<br>
    <strong>IBM AI Fairness 360 &nbsp;·&nbsp; IBM Granite-3.3-8b &nbsp;·&nbsp; watsonx.governance</strong>
  </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "🌍 **287,000 maternal deaths per year — 95% in low & middle-income countries. "
    "Most preventable if risk is caught early.** &nbsp;|&nbsp; "
    "UN SDG 3 · SDG 10 · IBM Z × UNSA Sheridan Hackathon 2026"
)

# ── Load & train ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training on WHO maternal health data…")
def load_and_train():
    df      = load_dataset()
    df_bin  = get_binary_df(df)
    results, best, X_tr, X_te, y_tr, y_te = train_models(df)
    return df, df_bin, results, best, X_tr, X_te, y_tr, y_te

df, df_bin, results, best_name, X_train, X_test, y_train, y_test = load_and_train()
best_model = results[best_name]["model"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=72)
    st.markdown("### IBM Tech Stack")
    st.success("✅ IBM AI Fairness 360")
    st.success("✅ IBM Granite-3.3-8b (watsonx)")
    st.success("✅ watsonx.governance architecture")
    granite_status = "✅ Granite API live" if is_live() else "⚙️ Add HF_TOKEN for live Granite"
    (st.success if is_live() else st.info)(granite_status)

    st.markdown("---")
    st.markdown("### Active Model")
    chosen = st.selectbox("Model", list(results.keys()),
                          index=list(results.keys()).index(best_name))
    model = results[chosen]["model"]

    st.markdown("---")
    st.markdown("### The Crisis")
    st.markdown("""
- **287,000** maternal deaths/year
- **95 %** in developing nations
- Teenage mothers face **3×** higher mortality
- Rural health workers lack diagnostic tools

**MaternaAI** puts WHO-grade risk stratification
in any health worker's hands — with IBM AI
ensuring **no mother is left behind by bias.**
    """)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Model Performance", "🏥 Live Risk Assessment", "⚖️ IBM Fairness Audit", "📊 Population Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("IBM watsonx AutoAI — Model Leaderboard")
    st.caption("Three models trained automatically. Ranked by macro AUC-ROC (one-vs-rest, 3-class).")

    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        is_best = name == best_name
        border  = "border:2px solid #0f62fe;" if is_best else "border:1px solid #393939;"
        with cols[i]:
            st.markdown(f"""
            <div class="risk-card" style="{border}background:#12171f;">
              <div style="color:#a6c8ff;font-weight:700">{name}</div>
              {"<div style='color:#42be65;font-size:.78em'>★ BEST MODEL</div>" if is_best else ""}
              <div style="font-size:2.4em;font-weight:800;color:white">{res['auc']:.3f}</div>
              <div style="color:#8d8d8d;font-size:.82em">Macro AUC-ROC</div>
              <div style="color:#a8a8a8;font-size:.82em">F1 {res['f1']:.3f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Feature Importance")
        fi = get_feature_importance(model)
        fi_df = pd.DataFrame([
            {"Feature": FEATURE_LABELS.get(k, k), "Importance": v}
            for k, v in fi.items()
        ])
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues",
                     template="plotly_dark")
        fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Classification Report")
        rep = results[chosen]["report"]
        rep_df = pd.DataFrame({
            "Class":     ["Low Risk", "Medium Risk", "High Risk", "Macro Avg"],
            "Precision": [rep["Low Risk"]["precision"], rep["Medium Risk"]["precision"],
                          rep["High Risk"]["precision"], rep["macro avg"]["precision"]],
            "Recall":    [rep["Low Risk"]["recall"],    rep["Medium Risk"]["recall"],
                          rep["High Risk"]["recall"],    rep["macro avg"]["recall"]],
            "F1":        [rep["Low Risk"]["f1-score"],  rep["Medium Risk"]["f1-score"],
                          rep["High Risk"]["f1-score"],  rep["macro avg"]["f1-score"]],
        }).set_index("Class")
        st.dataframe(rep_df.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None),
                     use_container_width=True)
        st.metric("Macro AUC-ROC", f"{results[chosen]['auc']:.3f}")
        st.metric("Macro F1", f"{results[chosen]['f1']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Risk Assessment
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🏥 Live Maternal Risk Assessment — IBM Granite Clinical Explanation")
    st.caption("Enter patient vitals. IBM Granite-3.3-8b generates a plain-language clinical brief for the health worker.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age    = st.slider("Age (years)", 10, 49, 22)
        sys_bp = st.slider("Systolic BP (mmHg)", 70, 160, 118)
    with c2:
        dia_bp = st.slider("Diastolic BP (mmHg)", 40, 100, 76)
        bs     = st.number_input("Blood Glucose (mmol/L)", 6.0, 19.0, 7.8, step=0.1)
    with c3:
        temp   = st.number_input("Body Temp (°F)", 96.0, 103.0, 98.6, step=0.1)
        hr     = st.slider("Heart Rate (bpm)", 50, 90, 72)

    vitals = {"Age": age, "SystolicBP": sys_bp, "DiastolicBP": dia_bp,
              "BS": bs, "BodyTemp": temp, "HeartRate": hr}

    if st.button("⚡ Assess Risk with IBM watsonx", type="primary", use_container_width=True):
        probs, pred = predict_risk(model, vitals)
        label       = RISK_LABEL[pred]
        color       = RISK_COLOR[pred]
        fi          = get_feature_importance(model)
        top_factors = list(fi.items())[:5]

        c_card, c_gauge = st.columns(2)
        with c_card:
            teen_flag = "⚠️ TEENAGE PATIENT — elevated bias risk" if age <= 19 else ""
            st.markdown(f"""
            <div class="risk-card" style="border-color:{color};background:#12171f;">
              <div style="font-size:3em;font-weight:800;color:{color}">{label}</div>
              <div style="color:#8d8d8d;margin:8px 0">Risk probabilities</div>
              <div>
                <span class="stat-pill" style="background:#152d15;color:#42be65">
                  Low {probs[0]:.0%}</span>
                <span class="stat-pill" style="background:#2d2515;color:#f1c21b">
                  Mid {probs[1]:.0%}</span>
                <span class="stat-pill" style="background:#2d1515;color:#fa4d56">
                  High {probs[2]:.0%}</span>
              </div>
              {"<div style='color:#f1c21b;margin-top:10px;font-size:.85em'>" + teen_flag + "</div>" if teen_flag else ""}
            </div>""", unsafe_allow_html=True)

        with c_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=probs[2] * 100,
                title={"text": "High-Risk Probability (%)", "font": {"color": "white"}},
                number={"suffix": "%", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": RISK_COLOR[pred]},
                    "steps": [{"range": [0, 33],  "color": "#152d15"},
                               {"range": [33, 66], "color": "#2d2515"},
                               {"range": [66, 100],"color": "#2d1515"}],
                    "threshold": {"line": {"color": "white", "width": 2},
                                  "thickness": 0.75, "value": 66},
                },
            ))
            fig.update_layout(template="plotly_dark", height=220,
                              margin=dict(t=30, b=0, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### IBM Granite-3.3-8b Clinical Brief")
        with st.spinner("IBM Granite generating clinical explanation…"):
            explanation = explain_risk(vitals, label, probs, top_factors)
        if pred == 2:
            st.error(explanation)
        elif pred == 1:
            st.warning(explanation)
        else:
            st.success(explanation)

        # Vital signs radar
        st.markdown("#### Patient Vital Signs vs Population Median")
        medians = df[FEATURE_COLS].median()
        categories = [FEATURE_LABELS[f] for f in FEATURE_COLS]
        patient_norm = [vitals[f] / medians[f] for f in FEATURE_COLS]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=patient_norm, theta=categories,
                                       fill="toself", name="This Patient",
                                       line_color=color))
        fig2.add_trace(go.Scatterpolar(r=[1]*len(FEATURE_COLS), theta=categories,
                                       fill="toself", name="Population Median",
                                       line_color="#6f6f6f", opacity=0.4))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True)),
                           template="plotly_dark", height=320,
                           margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IBM AIF360 Fairness Audit
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚖️ IBM AI Fairness 360 — Equity Audit")
    st.markdown("""
    **Sensitive attribute: Age group** — Teenage mothers (≤19) vs Adult mothers (20+).
    Teenage mothers face 3× higher mortality risk yet are systematically under-detected by biased AI.
    IBM AIF360 Reweighing mitigation is applied automatically.
    """)

    @st.cache_resource(show_spinner="Running IBM AI Fairness 360 audit…")
    def cached_audit(_model, _Xtr, _ytr, _Xte, _yte, _str, _ste):
        df_b    = get_binary_df(df)
        ytr_bin = df_b.loc[_Xtr.index, "label_binary"].values
        yte_bin = df_b.loc[_Xte.index, "label_binary"].values
        s_tr    = df_b.loc[_Xtr.index, "age_group"].values
        s_te    = df_b.loc[_Xte.index, "age_group"].values
        return run_audit(_model, _Xtr, ytr_bin, _Xte, yte_bin, s_tr, s_te, FEATURE_COLS)

    audit = cached_audit(model, X_train, y_train, X_test, y_test,
                         df.loc[X_train.index, "age_group"].values,
                         df.loc[X_test.index,  "age_group"].values)

    orig = audit.get("original", {})
    mit  = audit.get("mitigated", {})
    vd   = verdict(orig)

    badge = ("🟢 PASSES IBM AI FAIRNESS 360 STANDARDS" if vd["fair"]
             else "🔴 BIAS DETECTED — REWEIGHING MITIGATION APPLIED")
    st.markdown(f"### {badge}")

    m_keys = [
        ("Disparate Impact",              "disparate_impact",              "≥ 0.80", "di_pass"),
        ("Statistical Parity Diff.",      "statistical_parity_difference", "±0.10",  "spd_pass"),
        ("Equal Opportunity Diff.",       "equal_opportunity_difference",  "±0.10",  "eod_pass"),
        ("Average Odds Diff.",            "average_odds_difference",       "±0.10",  "aod_pass"),
    ]
    cols = st.columns(4)
    for i, (label, key, thresh, pass_key) in enumerate(m_keys):
        o = orig.get(key, 0)
        m_val = mit.get(key, o) if "error" not in mit else o
        with cols[i]:
            delta_str = f"→ {m_val:.3f} after Reweighing"
            st.metric(f"{label} ({thresh})", f"{o:.3f}", delta=delta_str)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Before vs After IBM AIF360 Reweighing")
        keys_plot   = [m[1] for m in m_keys]
        labels_plot = [m[0] for m in m_keys]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Original", x=labels_plot,
                             y=[orig.get(k, 0) for k in keys_plot],
                             marker_color="#fa4d56"))
        if "error" not in mit:
            fig.add_trace(go.Bar(name="After Reweighing (IBM AIF360)", x=labels_plot,
                                 y=[mit.get(k, 0) for k in keys_plot],
                                 marker_color="#42be65"))
        fig.add_hline(y=0.8, line_dash="dot", line_color="#f1c21b",
                      annotation_text="DI Fair Threshold")
        fig.add_hline(y=0.1, line_dash="dot", line_color="#a8a8a8",
                      annotation_text="+0.10 Threshold")
        fig.add_hline(y=-0.1, line_dash="dot", line_color="#a8a8a8")
        fig.update_layout(barmode="group", template="plotly_dark",
                          height=340, margin=dict(t=10, b=0),
                          legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### IBM Granite Governance Recommendation")
        with st.spinner("IBM Granite generating policy brief…"):
            policy = governance_policy(orig, mit)
        st.warning(policy)

        st.markdown("#### Fairness Scorecard")
        sc_df = pd.DataFrame([
            {"Metric": m[0], "Original": f"{orig.get(m[1],0):.3f}",
             "Mitigated": f"{mit.get(m[1],orig.get(m[1],0)):.3f}" if "error" not in mit else "N/A",
             "Threshold": m[2], "Status": "✅" if vd[m[3]] else "❌"}
            for m in m_keys
        ])
        st.dataframe(sc_df, use_container_width=True, hide_index=True)

    st.info(
        "**Why this matters:** An AI model biased against teenage mothers "
        "will under-detect their high-risk status — directly contributing to preventable deaths. "
        "IBM AIF360 makes this invisible inequity visible and correctable."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Population Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Maternal Health Population — WHO Dataset Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(df):,}")
    c2.metric("High Risk", f"{(df['label']==2).sum()} ({(df['label']==2).mean():.0%})")
    c3.metric("Medium Risk", f"{(df['label']==1).sum()} ({(df['label']==1).mean():.0%})")
    c4.metric("Teenage Mothers", f"{(df['Age']<=19).sum()} ({(df['Age']<=19).mean():.0%})")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Age",
                           color=df["label"].map(RISK_LABEL),
                           color_discrete_map={v: RISK_COLOR[k] for k, v in RISK_LABEL.items()},
                           nbins=30, template="plotly_dark",
                           title="Age Distribution by Risk Level",
                           labels={"color": "Risk Level"})
        fig.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(df, x="SystolicBP", y="BS",
                         color=df["label"].map(RISK_LABEL),
                         color_discrete_map={v: RISK_COLOR[k] for k, v in RISK_LABEL.items()},
                         template="plotly_dark", opacity=0.65,
                         title="Blood Pressure vs Blood Glucose by Risk Level",
                         labels={"SystolicBP": "Systolic BP (mmHg)",
                                 "BS": "Blood Glucose (mmol/L)",
                                 "color": "Risk Level"})
        fig.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        teen_risk = df[df["Age"] <= 19]["label"].value_counts().sort_index()
        adult_risk = df[df["Age"] >= 20]["label"].value_counts().sort_index()
        fig = go.Figure()
        for idx, name, color in [(0,"Low Risk","#42be65"),(1,"Mid Risk","#f1c21b"),(2,"High Risk","#fa4d56")]:
            fig.add_trace(go.Bar(
                name=name,
                x=["Teenage (≤19)", "Adult (≥20)"],
                y=[teen_risk.get(idx, 0) / max(len(df[df["Age"]<=19]),1),
                   adult_risk.get(idx, 0) / max(len(df[df["Age"]>=20]),1)],
                marker_color=color,
            ))
        fig.update_layout(barmode="stack", template="plotly_dark",
                          title="Risk Distribution: Teen vs Adult Mothers",
                          yaxis_tickformat=".0%", height=300,
                          margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(df, x=df["label"].map(RISK_LABEL), y="SystolicBP",
                     color=df["label"].map(RISK_LABEL),
                     color_discrete_map={v: RISK_COLOR[k] for k, v in RISK_LABEL.items()},
                     template="plotly_dark",
                     title="Systolic BP by Risk Level",
                     labels={"x": "Risk Level", "SystolicBP": "Systolic BP (mmHg)"})
        fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#6f6f6f;font-size:.82em">
  MaternaAI &nbsp;·&nbsp; IBM Z × UNSA Sheridan Hackathon 2026 &nbsp;·&nbsp;
  IBM AI Fairness 360 &nbsp;·&nbsp; IBM Granite-3.3-8b &nbsp;·&nbsp;
  UN SDG 3 Good Health &nbsp;·&nbsp; SDG 10 Reduced Inequalities
</div>
""", unsafe_allow_html=True)
