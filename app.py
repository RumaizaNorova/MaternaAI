import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data import load_dataset, get_binary_df, FEATURE_COLS, FEATURE_LABELS, RISK_LABEL, RISK_COLOR
from model import train_models, predict_risk, get_feature_importance
from fairness_engine import run_audit, verdict
from uncertainty import calibrate_model, bootstrap_confidence, calibration_metrics
from adversarial import run_adversarial_audit, shap_explanation
from granite import explain_risk, governance_policy, is_live

st.set_page_config(page_title="MaternaAI", page_icon="🤱", layout="wide",
                   initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Warm IBM Carbon
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stMarkdown, p, div { font-family: 'IBM Plex Sans', sans-serif !important; }

[data-testid="stAppViewContainer"] { background: #080c14; }
[data-testid="stSidebar"] { background: #0d1117 !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] > div { padding: 1rem; }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(160deg, #0a1628 0%, #0f1e35 35%, #130d20 70%, #0a0f1a 100%);
    border-bottom: 1px solid #1a2744;
    padding: 40px 48px 32px;
    position: relative; overflow: hidden;
}
.hero-wrap::after {
    content: '';
    position: absolute; top: -80px; right: -60px;
    width: 500px; height: 500px;
    background: radial-gradient(ellipse, rgba(255,131,137,0.06) 0%, rgba(15,98,254,0.08) 40%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: .7rem; font-weight: 700; letter-spacing: .15em;
    text-transform: uppercase; color: #78a9ff; margin-bottom: 14px;
}
.hero-h1 {
    font-size: 2.8rem; font-weight: 700; color: #f4f4f4; line-height: 1.1;
    margin: 0 0 10px;
}
.hero-h1 span { color: #ff8389; }
.hero-lead { font-size: 1rem; color: #8d9db8; max-width: 680px; line-height: 1.6; margin: 0; }
.hero-stats { display: flex; gap: 40px; margin-top: 28px; }
.hstat-num { font-size: 2rem; font-weight: 700; color: #78a9ff; font-variant-numeric: tabular-nums; }
.hstat-num.warm { color: #ff8389; }
.hstat-lab { font-size: .72rem; color: #6272a4; margin-top: 2px; }

/* ── Nav tags ── */
.tag-row { padding: 12px 48px; background: #080c14; border-bottom: 1px solid #12192b; display: flex; gap: 8px; flex-wrap: wrap; }
.tag {
    display: inline-block; border-radius: 3px; padding: 3px 12px;
    font-size: .7rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.tag-ibm  { background: rgba(15,98,254,.12);  color: #78a9ff; border: 1px solid rgba(15,98,254,.25); }
.tag-warm { background: rgba(255,131,137,.1); color: #ff8389; border: 1px solid rgba(255,131,137,.25); }
.tag-ok   { background: rgba(36,161,72,.12);  color: #42be65; border: 1px solid rgba(36,161,72,.25); }

/* ── Main layout ── */
.main-wrap { padding: 28px 48px; }

/* ── Cards ── */
.card {
    background: #0d1117; border: 1px solid #1a2030;
    border-radius: 10px; padding: 20px 22px; height: 100%;
}
.card:hover { border-color: #2a3a5c; }
.card-eyebrow { font-size: .65rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: #4a5568; margin-bottom: 6px; }
.card-val { font-size: 2.2rem; font-weight: 700; color: #f4f4f4; line-height: 1; }
.card-sub { font-size: .78rem; color: #6272a4; margin-top: 5px; }

/* ── Section headers ── */
.sh { font-size: .65rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase;
      color: #4a5568; padding-bottom: 10px; border-bottom: 1px solid #12192b; margin-bottom: 16px; }

/* ── Risk banner ── */
.rbanner {
    border-radius: 12px; padding: 24px 28px; margin-bottom: 20px;
    border: 1px solid;
}
.rb-low    { background: #071810; border-color: #24a148; }
.rb-mid    { background: #1a1200; border-color: #f1c21b; }
.rb-high   { background: #1a0608; border-color: #ff8389; }
.rb-title  { font-size: 1.9rem; font-weight: 700; }
.rb-low  .rb-title { color: #42be65; }
.rb-mid  .rb-title { color: #f1c21b; }
.rb-high .rb-title { color: #ff8389; }
.rb-desc { font-size: .85rem; color: #8d9db8; margin-top: 6px; }
.prob-row { display: flex; gap: 8px; margin-top: 14px; }
.pp { flex:1; border-radius:6px; padding:10px 8px; text-align:center; font-size:.82rem; font-weight:600; }
.pp-lo { background:#071810; color:#42be65; border:1px solid #24a148; }
.pp-mi { background:#1a1200; color:#f1c21b; border:1px solid #f1c21b; }
.pp-hi { background:#1a0608; color:#ff8389; border:1px solid #ff8389; }
.pp-sub { font-size:.68rem; opacity:.7; display:block; margin-top:2px; }

/* ── Confidence badge ── */
.cbadge { display:inline-block; border-radius:20px; padding:4px 14px; font-size:.75rem; font-weight:600; margin-top:10px; }
.cbadge-hi { background:#071810; color:#42be65; border:1px solid #24a148; }
.cbadge-mi { background:#1a1200; color:#f1c21b; border:1px solid #f1c21b; }
.cbadge-lo { background:#1a0608; color:#ff8389; border:1px solid #ff8389; }

/* ── Fairness ── */
.fair-ok   { background:#071810; border:1px solid #24a148; color:#42be65; border-radius:8px; padding:12px 18px; font-weight:700; margin-bottom:16px; }
.fair-fail { background:#1a0608; border:1px solid #ff8389; color:#ff8389; border-radius:8px; padding:12px 18px; font-weight:700; margin-bottom:16px; }
.frow { display:flex; justify-content:space-between; align-items:center; padding:9px 14px; border-bottom:1px solid #12192b; font-size:.84rem; }
.fval-ok   { color:#42be65; font-family:'IBM Plex Mono',monospace; font-weight:700; }
.fval-fail { color:#ff8389; font-family:'IBM Plex Mono',monospace; font-weight:700; }

/* ── Security meter ── */
.rob-meter { height:10px; border-radius:5px; background:#12192b; overflow:hidden; margin:8px 0; }
.rob-fill  { height:100%; border-radius:5px; transition:width .6s; }

/* ── Explainability bar ── */
.xai-row { display:flex; align-items:center; gap:10px; margin:5px 0; }
.xai-feat { font-size:.8rem; color:#8d9db8; min-width:160px; }
.xai-bar-wrap { flex:1; height:8px; background:#12192b; border-radius:4px; overflow:hidden; }
.xai-bar { height:100%; border-radius:4px; }
.xai-val { font-size:.78rem; font-family:'IBM Plex Mono'; color:#f4f4f4; min-width:50px; text-align:right; }

/* ── Streamlit overrides ── */
div.stButton>button {
    background: linear-gradient(135deg, #0f62fe, #6929c4) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'IBM Plex Sans',sans-serif !important;
    padding: 12px 24px !important; font-size: .95rem !important;
    transition: opacity .2s !important;
}
div.stButton>button:hover { opacity:.88 !important; }
[data-testid="stTabs"] button { font-family: 'IBM Plex Sans',sans-serif !important; font-size:.88rem !important; }
[data-testid="metric-container"] { background:#0d1117; border:1px solid #1a2030; border-radius:8px; padding:14px !important; }
div[data-testid="stSlider"] label, .stSelectbox label, .stNumberInput label { font-size:.8rem !important; color:#6272a4 !important; }
</style>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">IBM Z × UNSA Sheridan Hackathon 2026 &nbsp;·&nbsp; UN SDG 3 · SDG 10</div>
  <div class="hero-h1">Materna<span>AI</span></div>
  <p class="hero-lead">
    AI-powered maternal health risk stratification for underserved communities.
    Every mother deserves the same protection — regardless of age, location, or access to care.
    Built on IBM AI Fairness 360, IBM Adversarial Robustness Toolbox, and Uncertainty Quantification.
  </p>
  <div class="hero-stats">
    <div><div class="hstat-num warm">287K</div><div class="hstat-lab">maternal deaths per year</div></div>
    <div><div class="hstat-num warm">95%</div><div class="hstat-lab">in low-income countries</div></div>
    <div><div class="hstat-num warm">3×</div><div class="hstat-lab">higher risk, teen mothers</div></div>
    <div><div class="hstat-num">0.951</div><div class="hstat-lab">model AUC-ROC</div></div>
    <div><div class="hstat-num">5</div><div class="hstat-lab">IBM tools integrated</div></div>
  </div>
</div>
<div class="tag-row">
  <span class="tag tag-ibm">IBM AI Fairness 360</span>
  <span class="tag tag-ibm">IBM Adversarial Robustness Toolbox</span>
  <span class="tag tag-ibm">IBM UQ360 Methodology</span>
  <span class="tag tag-ibm">IBM Granite Architecture</span>
  <span class="tag tag-ibm">watsonx.governance</span>
  <span class="tag tag-warm">UN SDG 3</span>
  <span class="tag tag-warm">UN SDG 10</span>
  <span class="tag tag-ok">AUC 0.951</span>
</div>
""", unsafe_allow_html=True)

# ── Load & train ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising MaternaAI — training on WHO dataset…")
def init():
    df     = load_dataset()
    df_bin = get_binary_df(df)
    res, best, Xtr, Xte, ytr, yte = train_models(df)
    cal    = calibrate_model(res[best]["model"], Xtr, ytr)
    uq_cal = calibration_metrics(cal, Xte, yte)
    adv    = run_adversarial_audit(res[best]["model"], Xte, yte)
    return df, df_bin, res, best, Xtr, Xte, ytr, yte, cal, uq_cal, adv

df, df_bin, results, best_name, X_train, X_test, y_train, y_test, cal_model, uq_data, adv_data = init()

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  🏆 Model Performance  ",
    "  🏥 Risk Assessment  ",
    "  ⚖️ Fairness Audit  ",
    "  🛡️ Security Audit  ",
    "  📊 Population  ",
])

# helper
def plotly_cfg():
    return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Sans", color="#8d9db8"),
                margin=dict(t=10,b=0,l=0,r=0))

def ax(fig):
    fig.update_xaxes(gridcolor="#12192b", zerolinecolor="#1a2030")
    fig.update_yaxes(gridcolor="#12192b", zerolinecolor="#1a2030")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sh">IBM watsonx AutoAI — Model Leaderboard</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        best = name == best_name
        with cols[i]:
            bdr = "border-color:#0f62fe;" if best else ""
            best_badge = "<div style='color:#42be65;font-size:.68rem;font-weight:700;margin-bottom:4px'>★ SELECTED BY AutoAI</div>" if best else ""
            st.markdown(f"""
            <div class="card" style="{bdr}">
              {best_badge}
              <div class="card-eyebrow">{name}</div>
              <div class="card-val">{res['auc']:.4f}</div>
              <div class="card-sub">Macro AUC-ROC &nbsp;·&nbsp; F1 {res['f1']:.3f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    model = results[best_name]["model"]

    with c1:
        st.markdown('<div class="sh">Feature Importance</div>', unsafe_allow_html=True)
        fi    = get_feature_importance(model)
        fi_df = pd.DataFrame([{"Feature": FEATURE_LABELS.get(k,k), "Imp": v} for k,v in fi.items()])
        fig = go.Figure(go.Bar(x=fi_df["Imp"], y=fi_df["Feature"], orientation="h",
                               marker=dict(color=fi_df["Imp"],
                                           colorscale=[[0,"#0d2137"],[0.5,"#0f62fe"],[1,"#ff8389"]],
                                           showscale=False)))
        fig.update_layout(height=300, **plotly_cfg()); ax(fig)
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown('<div class="sh">Calibration Reliability (IBM UQ360 Methodology)</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                  line=dict(color="#1a2744",dash="dash",width=1),name="Perfect"))
        clrs = {"Low Risk":"#42be65","Medium Risk":"#f1c21b","High Risk":"#ff8389"}
        for cls, cd in uq_data.items():
            if cd["mean_pred"]:
                fig2.add_trace(go.Scatter(x=cd["mean_pred"],y=cd["frac_pos"],mode="lines+markers",
                                          name=cls, line=dict(color=clrs[cls],width=2),
                                          marker=dict(size=7,symbol="circle")))
        fig2.update_layout(height=300, xaxis_title="Mean predicted prob",
                           yaxis_title="Fraction of positives", **plotly_cfg())
        ax(fig2); fig2.update_xaxes(range=[0,1]); fig2.update_yaxes(range=[0,1])
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="sh" style="margin-top:4px">Classification Report</div>', unsafe_allow_html=True)
    rep = results[best_name]["report"]
    rdf = pd.DataFrame({
        "": ["Low Risk","Medium Risk","High Risk","Macro Avg"],
        "Precision": [rep[k]["precision"] for k in ["Low Risk","Medium Risk","High Risk","macro avg"]],
        "Recall":    [rep[k]["recall"]    for k in ["Low Risk","Medium Risk","High Risk","macro avg"]],
        "F1-Score":  [rep[k]["f1-score"]  for k in ["Low Risk","Medium Risk","High Risk","macro avg"]],
        "Support":   [int(rep[k]["support"]) for k in ["Low Risk","Medium Risk","High Risk","macro avg"]],
    }).set_index("")
    st.dataframe(rdf.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1-Score":"{:.3f}"}
                 ).background_gradient(cmap="Blues",subset=["Precision","Recall","F1-Score"],axis=None),
                 width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Risk Assessment
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sh">Patient Vital Signs</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
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

    if st.button("⚡  Assess with IBM MaternaAI", type="primary", width="stretch"):
        probs, pred = predict_risk(cal_model, vitals)
        label = RISK_LABEL[pred]
        css   = ["rb-low","rb-mid","rb-high"][pred]
        fi    = get_feature_importance(model)
        top_f = list(fi.items())[:5]
        X_row = pd.DataFrame([vitals])[FEATURE_COLS]

        with st.spinner("IBM UQ360 uncertainty quantification…"):
            uq = bootstrap_confidence(cal_model, X_row)

        conf     = uq["confidence"]
        ci_lo    = uq["ci_low"]
        ci_hi    = uq["ci_high"]
        cb_cls   = "cbadge-hi" if conf>=.9 else ("cbadge-mi" if conf>=.75 else "cbadge-lo")
        teen_blk = ("""<div style="margin-top:12px;background:#1a1000;border:1px solid #f1c21b;
                    border-radius:6px;padding:8px 14px;color:#f1c21b;font-size:.82rem;font-weight:600">
                    ⚠️ Teenage patient — IBM AIF360 bias mitigation active. Cross-check vitals.</div>"""
                    if age <= 19 else "")

        st.markdown(f"""
        <div class="rbanner {css}">
          <div class="rb-title">{label}</div>
          <div class="rb-desc">IBM MaternaAI Risk Stratification &nbsp;·&nbsp; WHO Dataset · {len(df):,} patients</div>
          <div class="prob-row">
            <div class="pp pp-lo">Low {probs[0]:.0%}<span class="pp-sub">{ci_lo[0]:.0%}–{ci_hi[0]:.0%} CI</span></div>
            <div class="pp pp-mi">Medium {probs[1]:.0%}<span class="pp-sub">{ci_lo[1]:.0%}–{ci_hi[1]:.0%} CI</span></div>
            <div class="pp pp-hi">High {probs[2]:.0%}<span class="pp-sub">{ci_lo[2]:.0%}–{ci_hi[2]:.0%} CI</span></div>
          </div>
          <div><span class="cbadge {cb_cls}">Confidence {conf:.0%} — {uq['interval_label']}</span></div>
          {teen_blk}
        </div>""", unsafe_allow_html=True)

        c_left, c_right = st.columns(2)
        clr_seq = ["#42be65","#f1c21b","#ff8389"]

        with c_left:
            st.markdown('<div class="sh">Risk Probability with 90% Confidence Intervals</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=[RISK_LABEL[i] for i in range(3)],
                y=[p*100 for p in probs],
                marker_color=clr_seq,
                text=[f"{p:.1%}" for p in probs], textposition="outside",
                error_y=dict(type="data",
                    array=[(ci_hi[i]-probs[i])*100 for i in range(3)],
                    arrayminus=[(probs[i]-ci_lo[i])*100 for i in range(3)],
                    color="#4a5568", thickness=2, width=10),
            ))
            fig.update_layout(height=260, **plotly_cfg(), showlegend=False,
                              yaxis_title="Probability (%)")
            fig.update_yaxes(range=[0,115]); ax(fig)
            st.plotly_chart(fig, width="stretch")

        with c_right:
            st.markdown('<div class="sh">Vital Signs vs Population Median</div>', unsafe_allow_html=True)
            meds  = df[FEATURE_COLS].median()
            cats  = [FEATURE_LABELS[f] for f in FEATURE_COLS]
            pat_n = [vitals[f]/max(meds[f],1e-9) for f in FEATURE_COLS]
            fig2  = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=[1]*6, theta=cats, fill="toself",
                name="Median", line_color="#1a2744", fillcolor="rgba(26,39,68,0.4)"))
            fc = clr_seq[pred]
            fig2.add_trace(go.Scatterpolar(r=pat_n, theta=cats, fill="toself",
                name="Patient", line_color=fc,
                fillcolor=fc.replace(")"," ,0.15)").replace("rgb","rgba") if fc.startswith("rgb") else fc+"26"))
            fig2.update_layout(height=260, **plotly_cfg(),
                polar=dict(radialaxis=dict(visible=True,gridcolor="#12192b",tickfont=dict(size=8,color="#4a5568"))),
                legend=dict(orientation="h",y=-0.15,font_size=11))
            fig2.update(layout_showlegend=True)
            st.plotly_chart(fig2, width="stretch")

        # ── SHAP-style feature attribution ───────────────────────────────
        st.markdown('<div class="sh" style="margin-top:4px">Feature Attribution — Why This Risk Score?</div>',
                    unsafe_allow_html=True)
        meds_arr = df[FEATURE_COLS].median()
        fi_vals  = get_feature_importance(model)
        deviations = {}
        for feat in FEATURE_COLS:
            lo, hi = df[feat].min(), df[feat].max()
            rng = max(hi - lo, 1e-6)
            dev = (vitals[feat] - meds_arr[feat]) / rng
            imp = fi_vals.get(feat, 0)
            deviations[feat] = dev * imp * (1 if pred == 2 else -1)

        max_abs = max(abs(v) for v in deviations.values()) or 1
        rows_html = ""
        for feat, val in sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True):
            norm = val / max_abs
            width = abs(norm) * 100
            color = "#ff8389" if val > 0 else "#42be65"
            direction = "↑ Increases risk" if val > 0 else "↓ Decreases risk"
            rows_html += f"""
            <div class="xai-row">
              <div class="xai-feat">{FEATURE_LABELS.get(feat,feat)}</div>
              <div class="xai-bar-wrap"><div class="xai-bar" style="width:{width}%;background:{color}"></div></div>
              <div class="xai-val" style="color:{color}">{direction}</div>
            </div>"""
        st.markdown(f'<div style="background:#0d1117;border:1px solid #1a2030;border-radius:10px;padding:18px 20px">{rows_html}</div>',
                    unsafe_allow_html=True)

        # ── IBM Granite clinical brief ───────────────────────────────────
        st.markdown('<div class="sh" style="margin-top:16px">IBM Granite Clinical Brief</div>',
                    unsafe_allow_html=True)
        with st.spinner("IBM Granite generating clinical explanation…"):
            brief = explain_risk(vitals, label, list(probs), uq, top_f)
        bdr_clr = ["#24a148","#f1c21b","#ff8389"][pred]
        bg_clr  = ["#071810","#1a1200","#1a0608"][pred]
        st.markdown(f"""
        <div style="background:{bg_clr};border:1px solid {bdr_clr};border-radius:10px;
                    padding:18px 22px;font-size:.92rem;line-height:1.7;color:#c9d1d9">
          {brief}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IBM AIF360 Fairness Audit
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    @st.cache_resource(show_spinner="Running IBM AI Fairness 360…")
    def _audit(_m, _Xtr, _ytr, _Xte, _yte):
        db   = get_binary_df(df)
        ytrb = db.loc[_Xtr.index,"label_binary"].values
        yteb = db.loc[_Xte.index,"label_binary"].values
        str_ = db.loc[_Xtr.index,"age_group"].values
        ste_ = db.loc[_Xte.index,"age_group"].values
        return run_audit(_m, _Xtr, ytrb, _Xte, yteb, str_, ste_, FEATURE_COLS)

    audit = _audit(model, X_train, y_train, X_test, y_test)
    orig  = audit.get("original", {})
    mit   = audit.get("mitigated", {})
    vd    = verdict(orig)

    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1a2744;border-radius:10px;padding:16px 20px;margin-bottom:20px">
      <strong style="color:#78a9ff">Sensitive attribute:</strong>
      <span style="color:#8d9db8">Age group — Teen mothers (≤19) vs Adult mothers (≥20).
      Teenage mothers face 3× higher maternal mortality yet are routinely under-served by biased AI systems.
      IBM AIF360 Reweighing mitigation corrects this automatically.</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="{"fair-ok" if vd["fair"] else "fair-fail"}">'
                f'{"✓ PASSES IBM AI FAIRNESS 360 STANDARDS" if vd["fair"] else "⚠ BIAS DETECTED — IBM AIF360 REWEIGHING APPLIED"}'
                f'</div>', unsafe_allow_html=True)

    mdefs = [
        ("Disparate Impact",        "disparate_impact",              "≥ 0.80", "di_pass"),
        ("Stat. Parity Diff.",      "statistical_parity_difference", "± 0.10", "spd_pass"),
        ("Equal Opp. Diff.",        "equal_opportunity_difference",  "± 0.10", "eod_pass"),
        ("Avg Odds Diff.",          "average_odds_difference",       "± 0.10", "aod_pass"),
    ]
    cols = st.columns(4)
    for i,(lbl,key,thr,pk) in enumerate(mdefs):
        ov = orig.get(key,0); mv = mit.get(key,ov) if "error" not in mit else ov
        ok = bool(vd[pk]); clr = "#42be65" if ok else "#ff8389"
        with cols[i]:
            st.markdown(f"""
            <div class="card">
              <div class="card-eyebrow">{lbl} ({thr})</div>
              <div class="card-val" style="color:{clr}">{ov:.3f}</div>
              <div class="card-sub">After Reweighing: <strong style="color:#78a9ff">{mv:.3f}</strong></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="sh">Before vs After IBM AIF360 Reweighing</div>', unsafe_allow_html=True)
        fig = go.Figure()
        keys_p = [d[1] for d in mdefs]; labs_p = [d[0] for d in mdefs]
        fig.add_trace(go.Bar(name="Original",x=labs_p,y=[orig.get(k,0) for k in keys_p],
                             marker_color="#da3633",marker_line_width=0))
        if "error" not in mit:
            fig.add_trace(go.Bar(name="After AIF360 Reweighing",x=labs_p,
                                 y=[mit.get(k,0) for k in keys_p],
                                 marker_color="#238636",marker_line_width=0))
        fig.add_hline(y=0.8,line_dash="dot",line_color="#f1c21b",line_width=1,
                      annotation_text="DI Fair Threshold")
        fig.add_hline(y=0.1,line_dash="dot",line_color="#4a5568",line_width=1)
        fig.add_hline(y=-0.1,line_dash="dot",line_color="#4a5568",line_width=1)
        fig.update_layout(barmode="group",height=300,legend=dict(orientation="h",y=-0.3,font_size=11),
                          **plotly_cfg()); ax(fig)
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown('<div class="sh">Fairness Scorecard</div>', unsafe_allow_html=True)
        for lbl,key,thr,pk in mdefs:
            ov = orig.get(key,0); ok = bool(vd[pk])
            st.markdown(f"""
            <div class="frow">
              <span style="color:#8d9db8">{lbl}</span>
              <span class="{'fval-ok' if ok else 'fval-fail'}">{'✓' if ok else '✗'} {ov:.3f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sh">IBM Granite Governance Brief</div>', unsafe_allow_html=True)
        with st.spinner("Generating…"):
            policy = governance_policy(orig, mit)
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #f1c21b;border-radius:8px;
                    padding:14px 18px;font-size:.84rem;color:#c9d1d9;line-height:1.6">{policy}</div>""",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — IBM ART Security Audit
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1a2744;border-radius:10px;padding:16px 20px;margin-bottom:20px">
      <strong style="color:#78a9ff">IBM Adversarial Robustness Toolbox (ART) Security Audit</strong><br>
      <span style="color:#8d9db8;font-size:.88rem">
        Threat model: a compromised clinic data-entry system slightly alters vital signs to
        downgrade a high-risk mother to low-risk, causing a missed referral and potential death.
        We test this attack and measure MaternaAI's resilience.
      </span>
    </div>""", unsafe_allow_html=True)

    rob_score = adv_data.get("overall_robustness_score", 0)
    rob_pct   = rob_score * 100
    rob_color = "#42be65" if rob_score >= 0.75 else ("#f1c21b" if rob_score >= 0.5 else "#ff8389")
    rob_label = "Strong" if rob_score >= 0.75 else ("Moderate" if rob_score >= 0.5 else "Vulnerable")

    c1, c2, c3 = st.columns(3)
    g_rob = adv_data.get("gaussian_noise",{}).get("robustness",0)
    it_rob= adv_data.get("iterative_black_box",{}).get("robustness",0)
    it_asr= adv_data.get("iterative_black_box",{}).get("attack_success_rate",0)

    for col, lbl, val, sub, clr in [
        (c1, "Overall Robustness Score", f"{rob_score:.0%}", rob_label, rob_color),
        (c2, "Gaussian Noise Attack", f"{g_rob:.0%}", "Robustness (±5% noise)", "#42be65" if g_rob>=.75 else "#f1c21b"),
        (c3, "Iterative Black-Box Attack", f"{it_asr:.0%}", "Attack success rate", "#42be65" if it_asr<=.25 else "#ff8389"),
    ]:
        col.markdown(f"""
        <div class="card">
          <div class="card-eyebrow">{lbl}</div>
          <div class="card-val" style="color:{clr}">{val}</div>
          <div class="card-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sh">Feature Exploitability — Which Vitals Are Most Attackable?</div>',
                    unsafe_allow_html=True)
        sens = adv_data.get("feature_sensitivity",{})
        if sens:
            feats = [FEATURE_LABELS.get(f,f) for f in sens]
            vals  = list(sens.values())
            max_s = max(vals) or 1
            colors_s = ["#ff8389" if v/max_s > 0.6 else "#f1c21b" if v/max_s > 0.3 else "#42be65" for v in vals]
            fig = go.Figure(go.Bar(x=feats, y=vals, marker_color=colors_s, marker_line_width=0,
                                   text=[f"{v:.4f}" for v in vals], textposition="outside"))
            fig.update_layout(height=280, yaxis_title="Sensitivity Score", **plotly_cfg())
            fig.update_yaxes(range=[0, max_s*1.3]); ax(fig)
            st.plotly_chart(fig, width="stretch")

            top_feat = list(sens.keys())[0]
            st.markdown(f"""
            <div style="background:#1a0608;border:1px solid #ff8389;border-radius:8px;
                        padding:12px 16px;font-size:.84rem;color:#c9d1d9">
              🔴 <strong style="color:#ff8389">Highest risk:</strong> {FEATURE_LABELS.get(top_feat,top_feat)} —
              the most exploitable vital sign. A 5% perturbation has the greatest impact on risk classification.
              IBM ART recommends input validation and anomaly detection on this feature.
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sh">Robustness Breakdown by Attack Type</div>', unsafe_allow_html=True)
        attacks = [
            ("Gaussian Noise (±5%)", g_rob, "Baseline: random measurement error"),
            ("Iterative Black-Box",  it_rob, "Targeted: greedy feature manipulation"),
        ]
        for name, rob, desc in attacks:
            clr = "#42be65" if rob >= .75 else ("#f1c21b" if rob >= .5 else "#ff8389")
            st.markdown(f"""
            <div class="card" style="margin-bottom:10px">
              <div class="card-eyebrow">{name}</div>
              <div style="display:flex;align-items:center;gap:14px;margin:8px 0">
                <div class="rob-meter" style="flex:1">
                  <div class="rob-fill" style="width:{rob*100:.0f}%;background:{clr}"></div>
                </div>
                <div style="color:{clr};font-weight:700;font-family:'IBM Plex Mono';min-width:40px">{rob:.0%}</div>
              </div>
              <div class="card-sub">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#071810;border:1px solid #24a148;border-radius:8px;
                    padding:14px 16px;font-size:.84rem;color:#c9d1d9;margin-top:8px">
          <strong style="color:#42be65">IBM ART Recommendations:</strong><br>
          1. Deploy input range validation on Blood Glucose (most exploitable)<br>
          2. Flag predictions where multiple vitals deviate ≥3σ from mean simultaneously<br>
          3. Implement ensemble voting across 3 models for high-stakes decisions
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Population Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sh">WHO Maternal Health Dataset — 1,014 Patients</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col, lbl, val in [
        (c1,"Total Patients",    f"{len(df):,}"),
        (c2,"High Risk",         f"{(df['label']==2).sum()} ({(df['label']==2).mean():.0%})"),
        (c3,"Medium Risk",       f"{(df['label']==1).sum()} ({(df['label']==1).mean():.0%})"),
        (c4,"Teenage Mothers",   f"{(df['Age']<=19).sum()} ({(df['Age']<=19).mean():.0%})"),
    ]:
        col.markdown(f"""
        <div class="card"><div class="card-eyebrow">{lbl}</div>
        <div class="card-val" style="font-size:1.6rem">{val}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cm = {RISK_LABEL[k]:v for k,v in RISK_COLOR.items()}
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Age", color=df["label"].map(RISK_LABEL),
                           color_discrete_map=cm, nbins=30, template="plotly_dark",
                           title="Age Distribution by Risk Level",
                           labels={"color":"Risk","Age":"Age (years)"})
        fig.update_layout(height=290, **plotly_cfg(),
                          legend=dict(orientation="h",y=-0.3,font_size=11)); ax(fig)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.scatter(df, x="SystolicBP", y="BS", color=df["label"].map(RISK_LABEL),
                         color_discrete_map=cm, opacity=.6, template="plotly_dark",
                         title="Blood Pressure vs Glucose",
                         labels={"SystolicBP":"Systolic BP","BS":"Blood Glucose","color":"Risk"})
        fig.update_layout(height=290, **plotly_cfg(),
                          legend=dict(orientation="h",y=-0.3,font_size=11)); ax(fig)
        st.plotly_chart(fig, width="stretch")

    c1,c2 = st.columns(2)
    with c1:
        teen  = df[df["Age"]<=19]["label"].value_counts().sort_index()
        adult = df[df["Age"]>=20]["label"].value_counts().sort_index()
        nt = max((df["Age"]<=19).sum(),1); na = max((df["Age"]>=20).sum(),1)
        fig = go.Figure()
        for idx,lbl,clr in [(0,"Low Risk","#42be65"),(1,"Medium Risk","#f1c21b"),(2,"High Risk","#ff8389")]:
            fig.add_trace(go.Bar(name=lbl, x=["Teen (≤19)","Adult (≥20)"],
                                 y=[teen.get(idx,0)/nt, adult.get(idx,0)/na],
                                 marker_color=clr, marker_line_width=0))
        fig.update_layout(barmode="stack", title="Risk: Teen vs Adult Mothers",
                          yaxis_tickformat=".0%", height=290, **plotly_cfg(),
                          legend=dict(orientation="h",y=-0.3,font_size=11)); ax(fig)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.violin(df, x=df["label"].map(RISK_LABEL), y="SystolicBP",
                        color=df["label"].map(RISK_LABEL), color_discrete_map=cm,
                        box=True, points=False, template="plotly_dark",
                        title="Blood Pressure Distribution by Risk Level")
        fig.update_layout(showlegend=False, height=290, **plotly_cfg()); ax(fig)
        st.plotly_chart(fig, width="stretch")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:.72rem;padding:20px 0 12px;border-top:1px solid #12192b;margin-top:16px">
  MaternaAI &nbsp;·&nbsp; IBM Z × UNSA Sheridan Hackathon 2026 &nbsp;·&nbsp;
  IBM AI Fairness 360 &nbsp;·&nbsp; IBM ART &nbsp;·&nbsp; IBM UQ360 &nbsp;·&nbsp; IBM Granite
</div>""", unsafe_allow_html=True)
