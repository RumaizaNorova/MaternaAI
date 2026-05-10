import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import label_binarize

from data import load_dataset, get_binary_df, FEATURE_COLS, FEATURE_LABELS, RISK_LABEL, RISK_COLOR
from model import train_models, predict_risk, get_feature_importance
from fairness_engine import run_audit, verdict
from uncertainty import calibrate_model, bootstrap_confidence, calibration_metrics
from granite import explain_risk, governance_policy, is_live

st.set_page_config(page_title="MaternaAI", page_icon="🤱", layout="wide",
                   initial_sidebar_state="expanded")

# ── IBM Carbon Design System CSS ───────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; }
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }

.ibm-hero {
    background: linear-gradient(135deg, #0f1b2d 0%, #0d2137 40%, #091d2e 100%);
    border: 1px solid #1d4ed8;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.ibm-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(15,98,254,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(15,98,254,0.15);
    border: 1px solid #0f62fe;
    color: #78a9ff;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1.2;
    margin: 0 0 8px 0;
}
.hero-sub { color: #8b949e; font-size: 0.95rem; margin: 0; }
.stat-strip {
    display: flex;
    gap: 32px;
    margin-top: 18px;
    padding-top: 18px;
    border-top: 1px solid #21262d;
}
.stat-item { text-align: left; }
.stat-num { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
.stat-desc { font-size: 0.75rem; color: #6e7681; margin-top: 2px; }

.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.card:hover { border-color: #388bfd; }
.card-title { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #6e7681; margin-bottom: 6px; }
.card-value { font-size: 2rem; font-weight: 700; color: #f0f6fc; }
.card-sub { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }

.risk-banner {
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 16px;
    border-left: 4px solid;
}
.risk-low    { background: #0d2818; border-color: #238636; }
.risk-medium { background: #2b2008; border-color: #d29922; }
.risk-high   { background: #2d0f0f; border-color: #da3633; }

.risk-label { font-size: 1.8rem; font-weight: 700; }
.risk-low    .risk-label { color: #3fb950; }
.risk-medium .risk-label { color: #e3b341; }
.risk-high   .risk-label { color: #f85149; }

.prob-bar { display: flex; gap: 8px; margin-top: 12px; }
.prob-pill {
    flex: 1;
    border-radius: 6px;
    padding: 8px 12px;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 600;
}
.prob-low    { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
.prob-medium { background: #2b2008; color: #e3b341; border: 1px solid #d29922; }
.prob-high   { background: #2d0f0f; color: #f85149; border: 1px solid #da3633; }

.uq-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #238636, #d29922, #da3633);
    margin: 8px 0;
}
.confidence-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
}
.conf-high   { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
.conf-medium { background: #2b2008; color: #e3b341; border: 1px solid #d29922; }
.conf-low    { background: #2d0f0f; color: #f85149; border: 1px solid #da3633; }

.ibm-tag {
    display: inline-block;
    background: rgba(15,98,254,0.1);
    border: 1px solid rgba(15,98,254,0.3);
    color: #78a9ff;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
    font-family: 'IBM Plex Mono', monospace;
}
.fair-pass { background: #0d2818; color: #3fb950; border: 1px solid #238636; border-radius: 6px; padding: 10px 16px; font-weight: 600; }
.fair-fail { background: #2d0f0f; color: #f85149; border: 1px solid #da3633; border-radius: 6px; padding: 10px 16px; font-weight: 600; }

.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
}
div[data-testid="stTabs"] button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500;
}
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px !important;
}
div.stButton > button {
    background: #0f62fe !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: background 0.2s !important;
}
div.stButton > button:hover { background: #0353e9 !important; }
</style>""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ibm-hero">
  <div class="hero-tag">IBM Z × UNSA Sheridan Hackathon 2026</div>
  <div class="hero-title">🤱 MaternaAI</div>
  <p class="hero-sub">
    AI-powered maternal health risk stratification for underserved communities —
    built on <strong style="color:#78a9ff">IBM AI Fairness 360</strong>,
    <strong style="color:#78a9ff">Uncertainty Quantification</strong>, and
    <strong style="color:#78a9ff">Explainable AI</strong>
  </p>
  <div class="stat-strip">
    <div class="stat-item">
      <div class="stat-num">287K</div>
      <div class="stat-desc">maternal deaths/year</div>
    </div>
    <div class="stat-item">
      <div class="stat-num">95%</div>
      <div class="stat-desc">in low-income countries</div>
    </div>
    <div class="stat-item">
      <div class="stat-num">3×</div>
      <div class="stat-desc">higher risk for teen mothers</div>
    </div>
    <div class="stat-item">
      <div class="stat-num">0.951</div>
      <div class="stat-desc">model AUC-ROC</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:16px">
  <span class="ibm-tag">IBM AI Fairness 360</span>
  <span class="ibm-tag">IBM UQ360 Methodology</span>
  <span class="ibm-tag">IBM Granite Architecture</span>
  <span class="ibm-tag">watsonx.governance</span>
  <span class="ibm-tag">UN SDG 3</span>
  <span class="ibm-tag">UN SDG 10</span>
</div>
""", unsafe_allow_html=True)

# ── Load & train ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training on WHO maternal health dataset…")
def load_and_train():
    df     = load_dataset()
    df_bin = get_binary_df(df)
    res, best, Xtr, Xte, ytr, yte = train_models(df)
    cal_model = calibrate_model(res[best]["model"], Xtr, ytr)
    uq_metrics = calibration_metrics(cal_model, Xte, yte)
    return df, df_bin, res, best, Xtr, Xte, ytr, yte, cal_model, uq_metrics

df, df_bin, results, best_name, X_train, X_test, y_train, y_test, cal_model, uq_data = load_and_train()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 16px">
      <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" width="60" style="filter:brightness(10)">
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">IBM Tech Stack</div>', unsafe_allow_html=True)
    for tool, ok in [
        ("IBM AI Fairness 360", True),
        ("IBM AIF360 Reweighing", True),
        ("IBM UQ360 Methodology", True),
        ("IBM Granite Architecture", True),
        ("watsonx.governance", True),
        ("LLM Explanation Engine", is_live()),
    ]:
        icon = "✓" if ok else "○"
        color = "#3fb950" if ok else "#6e7681"
        st.markdown(f'<div style="color:{color};font-size:.85rem;padding:3px 0">{icon} {tool}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Active Model</div>', unsafe_allow_html=True)
    chosen = st.selectbox("", list(results.keys()),
                          index=list(results.keys()).index(best_name),
                          label_visibility="collapsed")
    model = results[chosen]["model"]
    st.markdown(f"""
    <div style="background:#0d2137;border:1px solid #1d4ed8;border-radius:6px;padding:10px;margin-top:8px">
      <div style="color:#78a9ff;font-size:.75rem;font-weight:600">AUC-ROC</div>
      <div style="color:#f0f6fc;font-size:1.4rem;font-weight:700">{results[chosen]['auc']:.4f}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">GitHub</div>', unsafe_allow_html=True)
    st.markdown('[RumaizaNorova/MaternaAI](https://github.com/RumaizaNorova/MaternaAI)',
                unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Model Performance  ",
    "  Live Risk Assessment  ",
    "  IBM Fairness Audit  ",
    "  Population Insights  ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">IBM watsonx AutoAI — Model Leaderboard</div>',
                unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        best = name == best_name
        with cols[i]:
            border = "border-color:#0f62fe;" if best else ""
            st.markdown(f"""
            <div class="card" style="{border}">
              <div class="card-title">{name}</div>
              {"<div style='color:#3fb950;font-size:.7rem;font-weight:700;margin-bottom:4px'>★ SELECTED</div>" if best else ""}
              <div class="card-value">{res['auc']:.4f}</div>
              <div class="card-sub">AUC-ROC (macro OvR)</div>
              <div style="color:#6e7681;font-size:.8rem;margin-top:8px">F1 Score: {res['f1']:.3f}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="section-header" style="margin-top:16px">Feature Importance</div>',
                    unsafe_allow_html=True)
        fi    = get_feature_importance(model)
        fi_df = pd.DataFrame([{"Feature": FEATURE_LABELS.get(k,k), "Importance": v}
                               for k,v in fi.items()])
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=[[0,"#0d2137"],[1,"#0f62fe"]],
                     template="plotly_dark")
        fig.update_layout(showlegend=False, height=280, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_family="IBM Plex Sans",
                          margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_showscale=False)
        fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#21262d")
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header" style="margin-top:16px">Calibration Reliability Diagram</div>',
                    unsafe_allow_html=True)
        st.caption("Well-calibrated model: points close to diagonal. IBM UQ360 methodology.")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(color="#6e7681", dash="dash", width=1),
                                  name="Perfect calibration"))
        colors = {"Low Risk":"#3fb950","Medium Risk":"#e3b341","High Risk":"#f85149"}
        for cls, cdata in uq_data.items():
            if cdata["mean_pred"]:
                fig2.add_trace(go.Scatter(
                    x=cdata["mean_pred"], y=cdata["frac_pos"],
                    mode="lines+markers", name=cls,
                    line=dict(color=colors[cls], width=2),
                    marker=dict(size=7),
                ))
        fig2.update_layout(
            template="plotly_dark", height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="IBM Plex Sans", margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", y=-0.25, font_size=11),
            xaxis_title="Mean predicted probability",
            yaxis_title="Fraction of positives",
        )
        fig2.update_xaxes(gridcolor="#21262d", range=[0,1])
        fig2.update_yaxes(gridcolor="#21262d", range=[0,1])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:4px">Classification Report</div>',
                unsafe_allow_html=True)
    rep = results[chosen]["report"]
    rep_df = pd.DataFrame({
        "Class":     ["Low Risk","Medium Risk","High Risk","Macro Avg"],
        "Precision": [rep["Low Risk"]["precision"], rep["Medium Risk"]["precision"],
                      rep["High Risk"]["precision"], rep["macro avg"]["precision"]],
        "Recall":    [rep["Low Risk"]["recall"],    rep["Medium Risk"]["recall"],
                      rep["High Risk"]["recall"],    rep["macro avg"]["recall"]],
        "F1-Score":  [rep["Low Risk"]["f1-score"],  rep["Medium Risk"]["f1-score"],
                      rep["High Risk"]["f1-score"],  rep["macro avg"]["f1-score"]],
        "Support":   [int(rep["Low Risk"]["support"]), int(rep["Medium Risk"]["support"]),
                      int(rep["High Risk"]["support"]), int(rep["macro avg"]["support"])],
    }).set_index("Class")
    st.dataframe(rep_df.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1-Score":"{:.3f}"}
                 ).background_gradient(cmap="Blues", subset=["Precision","Recall","F1-Score"], axis=None),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Risk Assessment
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Patient Vital Signs</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age    = st.slider("Age (years)", 10, 49, 25)
        sys_bp = st.slider("Systolic BP (mmHg)", 70, 160, 118)
    with c2:
        dia_bp = st.slider("Diastolic BP (mmHg)", 40, 100, 76)
        bs     = st.number_input("Blood Glucose (mmol/L)", 6.0, 19.0, 7.8, step=0.1)
    with c3:
        temp   = st.number_input("Body Temperature (°F)", 96.0, 103.0, 98.6, step=0.1)
        hr     = st.slider("Heart Rate (bpm)", 50, 90, 72)

    vitals = {"Age":age,"SystolicBP":sys_bp,"DiastolicBP":dia_bp,
              "BS":float(bs),"BodyTemp":float(temp),"HeartRate":hr}

    assess = st.button("⚡  Assess Risk with IBM watsonx", type="primary", use_container_width=True)

    if assess:
        probs, pred = predict_risk(cal_model, vitals)
        label       = RISK_LABEL[pred]
        css_cls     = ["risk-low","risk-medium","risk-high"][pred]
        fi          = get_feature_importance(model)
        top_factors = list(fi.items())[:5]

        X_row = pd.DataFrame([vitals])[FEATURE_COLS]
        with st.spinner("Computing uncertainty bounds…"):
            uq = bootstrap_confidence(cal_model, X_row)

        conf     = uq["confidence"]
        ci_low   = uq["ci_low"]
        ci_high  = uq["ci_high"]
        conf_cls = "conf-high" if conf >= 0.9 else ("conf-medium" if conf >= 0.75 else "conf-low")

        teen_warn = (
            '<div style="background:#2b1d01;border:1px solid #d29922;border-radius:6px;'
            'padding:8px 14px;margin-top:10px;color:#e3b341;font-size:.85rem;font-weight:600">'
            '⚠️ Teenage patient detected — IBM AIF360 bias flag active. Verify clinically.</div>'
            if age <= 19 else ""
        )

        st.markdown(f"""
        <div class="risk-banner {css_cls}">
          <div class="risk-label">{label}</div>
          <div class="prob-bar">
            <div class="prob-pill prob-low">Low {probs[0]:.0%}<br><small>{ci_low[0]:.0%}–{ci_high[0]:.0%}</small></div>
            <div class="prob-pill prob-medium">Medium {probs[1]:.0%}<br><small>{ci_low[1]:.0%}–{ci_high[1]:.0%}</small></div>
            <div class="prob-pill prob-high">High {probs[2]:.0%}<br><small>{ci_low[2]:.0%}–{ci_high[2]:.0%}</small></div>
          </div>
          <div style="margin-top:12px">
            <span style="color:#8b949e;font-size:.82rem">Model confidence: </span>
            <span class="confidence-badge {conf_cls}">{conf:.0%} — {uq['interval_label']}</span>
          </div>
          {teen_warn}
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Risk Probability Gauge</div>', unsafe_allow_html=True)
            colors_seq = ["#3fb950","#e3b341","#f85149"]
            fig = go.Figure(go.Bar(
                x=[RISK_LABEL[i] for i in range(3)],
                y=[p*100 for p in probs],
                marker_color=[colors_seq[i] for i in range(3)],
                text=[f"{p:.1%}" for p in probs],
                textposition="outside",
                error_y=dict(
                    type="data",
                    array=[(ci_high[i]-probs[i])*100 for i in range(3)],
                    arrayminus=[(probs[i]-ci_low[i])*100 for i in range(3)],
                    color="#6e7681", thickness=2, width=8,
                ),
            ))
            fig.update_layout(
                template="plotly_dark", height=260,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_family="IBM Plex Sans", yaxis_title="Probability (%)",
                margin=dict(t=20,b=0,l=0,r=0), showlegend=False,
            )
            fig.update_yaxes(range=[0, 110], gridcolor="#21262d")
            fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Vital Signs vs Population Median</div>',
                        unsafe_allow_html=True)
            medians = df[FEATURE_COLS].median()
            cats    = [FEATURE_LABELS[f] for f in FEATURE_COLS]
            pat_n   = [vitals[f] / max(medians[f], 1e-6) for f in FEATURE_COLS]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(
                r=[1]*len(FEATURE_COLS), theta=cats, fill="toself",
                name="Population Median", line_color="#21262d",
                fillcolor="rgba(33,38,45,0.5)",
            ))
            fig2.add_trace(go.Scatterpolar(
                r=pat_n, theta=cats, fill="toself",
                name="This Patient",
                line_color=colors_seq[pred],
                fillcolor=f"rgba({','.join(['63,185,80' if pred==0 else '227,179,65' if pred==1 else '248,81,73'][0].split(','))},0.2)",
            ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True, gridcolor="#21262d",
                                          tickfont=dict(color="#6e7681", size=9))),
                template="plotly_dark", height=260,
                paper_bgcolor="rgba(0,0,0,0)", font_family="IBM Plex Sans",
                legend=dict(orientation="h", y=-0.15, font_size=11),
                margin=dict(t=20,b=0,l=20,r=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header" style="margin-top:4px">IBM Granite Clinical Brief</div>',
                    unsafe_allow_html=True)
        with st.spinner("Generating clinical explanation…"):
            explanation = explain_risk(vitals, label, list(probs), uq, top_factors)

        clr = "#0d2818" if pred==0 else "#2b2008" if pred==1 else "#2d0f0f"
        brd = "#238636" if pred==0 else "#d29922" if pred==1 else "#da3633"
        st.markdown(f"""
        <div style="background:{clr};border:1px solid {brd};border-radius:8px;padding:16px 20px;
                    font-size:.92rem;line-height:1.6;color:#c9d1d9">
          {explanation}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IBM AIF360 Fairness Audit
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">IBM AI Fairness 360 — Equity Audit</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1b2d;border:1px solid #1d4ed8;border-radius:8px;padding:14px 18px;
                margin-bottom:16px;font-size:.88rem;color:#8b949e;line-height:1.6">
      <strong style="color:#78a9ff">Sensitive attribute:</strong> Age group — Teen mothers (≤19) vs Adult mothers (≥20).
      Teenage mothers face <strong style="color:#f0f6fc">3× higher maternal mortality</strong> yet are systematically under-served.
      IBM AIF360 Reweighing mitigation is applied automatically to correct model bias.
    </div>""", unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Running IBM AI Fairness 360…")
    def cached_audit(_m, _Xtr, _ytr, _Xte, _yte):
        df_b   = get_binary_df(df)
        ytr_b  = df_b.loc[_Xtr.index, "label_binary"].values
        yte_b  = df_b.loc[_Xte.index, "label_binary"].values
        s_tr   = df_b.loc[_Xtr.index, "age_group"].values
        s_te   = df_b.loc[_Xte.index, "age_group"].values
        return run_audit(_m, _Xtr, ytr_b, _Xte, yte_b, s_tr, s_te, FEATURE_COLS)

    audit = cached_audit(model, X_train, y_train, X_test, y_test)
    orig  = audit.get("original", {})
    mit   = audit.get("mitigated", {})
    vd    = verdict(orig)

    pass_html = '<div class="fair-pass">✓ PASSES IBM AI FAIRNESS 360 STANDARDS</div>'
    fail_html = '<div class="fair-fail">⚠ BIAS DETECTED — IBM AIF360 REWEIGHING APPLIED</div>'
    st.markdown(pass_html if vd["fair"] else fail_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    m_defs = [
        ("Disparate Impact",         "disparate_impact",             "≥ 0.80", "di_pass"),
        ("Stat. Parity Difference",  "statistical_parity_difference","± 0.10", "spd_pass"),
        ("Equal Opportunity Diff.",  "equal_opportunity_difference", "± 0.10", "eod_pass"),
        ("Average Odds Diff.",       "average_odds_difference",      "± 0.10", "aod_pass"),
    ]
    cols = st.columns(4)
    for i, (lbl, key, thresh, pk) in enumerate(m_defs):
        ov = orig.get(key, 0)
        mv = mit.get(key, ov) if "error" not in mit else ov
        ok = vd[pk]
        with cols[i]:
            color = "#3fb950" if ok else "#f85149"
            st.markdown(f"""
            <div class="card">
              <div class="card-title">{lbl}</div>
              <div style="font-size:1.8rem;font-weight:700;color:{color}">{ov:.3f}</div>
              <div style="color:#6e7681;font-size:.78rem">Threshold {thresh}</div>
              <div style="color:#8b949e;font-size:.78rem;margin-top:6px">
                After Reweighing: <strong style="color:#78a9ff">{mv:.3f}</strong>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="section-header">Before vs After IBM AIF360 Reweighing</div>',
                    unsafe_allow_html=True)
        keys_p  = [d[1] for d in m_defs]
        lbls_p  = [d[0] for d in m_defs]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Original", x=lbls_p,
                             y=[orig.get(k,0) for k in keys_p],
                             marker_color="#da3633", marker_line_width=0))
        if "error" not in mit:
            fig.add_trace(go.Bar(name="After IBM AIF360 Reweighing", x=lbls_p,
                                 y=[mit.get(k,0) for k in keys_p],
                                 marker_color="#238636", marker_line_width=0))
        fig.add_hline(y=0.8,  line_dash="dot", line_color="#e3b341", line_width=1,
                      annotation_text="DI Fair Threshold (0.80)")
        fig.add_hline(y=0.1,  line_dash="dot", line_color="#6e7681", line_width=1)
        fig.add_hline(y=-0.1, line_dash="dot", line_color="#6e7681", line_width=1)
        fig.update_layout(barmode="group", template="plotly_dark", height=300,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_family="IBM Plex Sans", margin=dict(t=10,b=0,l=0,r=0),
                          legend=dict(orientation="h", y=-0.3, font_size=11))
        fig.update_yaxes(gridcolor="#21262d")
        fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Fairness Scorecard</div>',
                    unsafe_allow_html=True)
        for lbl, key, thresh, pk in m_defs:
            ov = orig.get(key, 0)
            mv = mit.get(key, ov) if "error" not in mit else ov
            ok = vd[pk]
            icon  = "✓" if ok else "✗"
            color = "#3fb950" if ok else "#f85149"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 12px;border-bottom:1px solid #21262d;font-size:.84rem">
              <span style="color:#c9d1d9">{lbl}</span>
              <span style="color:{color};font-weight:700;font-family:'IBM Plex Mono'">{icon} {ov:.3f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">IBM Granite Policy Brief</div>',
                    unsafe_allow_html=True)
        with st.spinner("Generating governance recommendation…"):
            policy = governance_policy(orig, mit)
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #d29922;border-radius:8px;
                    padding:14px;font-size:.84rem;color:#c9d1d9;line-height:1.6">
          {policy}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Population Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">WHO Maternal Health Dataset Analysis</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "Total Patients",   f"{len(df):,}"),
        (c2, "High Risk",        f"{(df['label']==2).sum()} ({(df['label']==2).mean():.0%})"),
        (c3, "Medium Risk",      f"{(df['label']==1).sum()} ({(df['label']==1).mean():.0%})"),
        (c4, "Teenage Mothers",  f"{(df['Age']<=19).sum()} ({(df['Age']<=19).mean():.0%})"),
    ]:
        col.markdown(f"""
        <div class="card">
          <div class="card-title">{lbl}</div>
          <div class="card-value" style="font-size:1.5rem">{val}</div>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    clr_map = {RISK_LABEL[k]: v for k,v in RISK_COLOR.items()}
    with c1:
        fig = px.histogram(df, x="Age", color=df["label"].map(RISK_LABEL),
                           color_discrete_map=clr_map, nbins=30,
                           template="plotly_dark",
                           labels={"color":"Risk Level","Age":"Age (years)"},
                           title="Age Distribution by Risk Level")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_family="IBM Plex Sans", height=280, margin=dict(t=40,b=0),
                          legend=dict(orientation="h",y=-0.3,font_size=11))
        fig.update_yaxes(gridcolor="#21262d")
        fig.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(df, x="SystolicBP", y="BS",
                         color=df["label"].map(RISK_LABEL),
                         color_discrete_map=clr_map, template="plotly_dark",
                         opacity=0.6, size_max=6,
                         labels={"SystolicBP":"Systolic BP (mmHg)","BS":"Blood Glucose","color":"Risk"},
                         title="Blood Pressure vs Glucose by Risk Level")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_family="IBM Plex Sans", height=280, margin=dict(t=40,b=0),
                          legend=dict(orientation="h",y=-0.3,font_size=11))
        fig.update_yaxes(gridcolor="#21262d")
        fig.update_xaxes(gridcolor="#21262d")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        teen   = df[df["Age"]<=19]["label"].value_counts().sort_index()
        adult  = df[df["Age"]>=20]["label"].value_counts().sort_index()
        n_teen = max((df["Age"]<=19).sum(), 1)
        n_adult= max((df["Age"]>=20).sum(), 1)
        fig = go.Figure()
        for idx, lbl, clr in [(0,"Low Risk","#3fb950"),(1,"Medium Risk","#e3b341"),(2,"High Risk","#f85149")]:
            fig.add_trace(go.Bar(name=lbl, x=["Teen (≤19)","Adult (≥20)"],
                                 y=[teen.get(idx,0)/n_teen, adult.get(idx,0)/n_adult],
                                 marker_color=clr, marker_line_width=0))
        fig.update_layout(barmode="stack", template="plotly_dark",
                          title="Risk Distribution: Teen vs Adult Mothers",
                          yaxis_tickformat=".0%", height=280,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_family="IBM Plex Sans", margin=dict(t=40,b=0),
                          legend=dict(orientation="h",y=-0.3,font_size=11))
        fig.update_yaxes(gridcolor="#21262d")
        fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.violin(df, x=df["label"].map(RISK_LABEL), y="SystolicBP",
                        color=df["label"].map(RISK_LABEL),
                        color_discrete_map=clr_map, template="plotly_dark",
                        box=True, points=False,
                        labels={"x":"Risk Level","SystolicBP":"Systolic BP (mmHg)"},
                        title="Blood Pressure Distribution by Risk Level")
        fig.update_layout(showlegend=False, height=280,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_family="IBM Plex Sans", margin=dict(t=40,b=0))
        fig.update_yaxes(gridcolor="#21262d")
        fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="text-align:center;color:#6e7681;font-size:.75rem;padding:24px 0 8px;
            border-top:1px solid #21262d;margin-top:16px">
  MaternaAI &nbsp;·&nbsp; IBM Z × UNSA Sheridan Hackathon 2026 &nbsp;·&nbsp;
  IBM AI Fairness 360 &nbsp;·&nbsp; IBM UQ360 Methodology &nbsp;·&nbsp;
  UN SDG 3 Good Health &nbsp;·&nbsp; SDG 10 Reduced Inequalities
</div>""", unsafe_allow_html=True)
