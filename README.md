# MaternaAI

**AI-powered maternal health risk stratification for underserved communities**

*IBM Z × UNSA Sheridan Hackathon 2026 — UN SDG 3 · SDG 10*

---

## The Problem

**287,000 mothers die every year from preventable pregnancy complications. 95% in low-income countries. Teenage mothers face 3× higher mortality — yet are routinely misclassified or ignored by biased clinical AI systems.**

A healthcare worker in rural Bangladesh or rural Uganda sees dozens of patients with no specialist support. They need a decision tool that is:
- **Accurate** — catches high-risk mothers before a crisis
- **Fair** — doesn't systematically miss teenage patients
- **Trustworthy** — quantifies its own uncertainty so clinicians know when to escalate
- **Secure** — resilient to data manipulation at the point of entry

MaternaAI addresses all four.

---

## IBM Tools Integrated (5)

| Tool | What We Use It For |
|------|-------------------|
| **IBM AI Fairness 360** | Bias audit (teen vs adult mothers) + Reweighing mitigation |
| **IBM Adversarial Robustness Toolbox (ART)** | Security audit — adversarial vital-sign manipulation attacks |
| **IBM UQ360 Methodology** | Bootstrap uncertainty quantification + calibration reliability |
| **IBM Granite Architecture** | LLM clinical explanation engine (3-sentence actionable briefs) |
| **watsonx.governance** | Bias monitoring framework + audit trail design |

---

## Technical Architecture

```
WHO Maternal Health Dataset (UCI, 1,014 patients)
        │
        ▼
┌─────────────────────┐
│   data.py           │  Feature engineering, age_group encoding
│   FEATURE_COLS × 6  │  Age · SystolicBP · DiastolicBP · BS · BodyTemp · HeartRate
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│   model.py          │     │   IBM AI Fairness 360         │
│   3 models trained  │────▶│   fairness_engine.py          │
│   GBM · RF · LR     │     │   DI · SPD · EOD · AOD audit  │
│   Best: GBM 0.951   │     │   + Reweighing mitigation     │
└────────┬────────────┘     └──────────────────────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│   uncertainty.py    │     │   IBM ART                    │
│   CalibratedCV      │     │   adversarial.py              │
│   Bootstrap 200×    │────▶│   Gaussian noise attack       │
│   90% CI intervals  │     │   Iterative black-box attack  │
│   Stability score   │     │   Feature sensitivity map     │
└────────┬────────────┘     └──────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   granite.py        │  IBM Granite LLM → 3-sentence clinical brief
│   WHO guidelines    │  Risk · Clinical concern · Immediate action
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   app.py (Streamlit)│  IBM Carbon Design System UI
│   5 tabs            │  Model · Risk · Fairness · Security · Population
└─────────────────────┘
```

### Model Performance
| Model | AUC-ROC (Macro OvR) | F1 (Macro) |
|-------|--------------------|----|
| **Gradient Boosting** *(selected)* | **0.9513** | **0.843** |
| Random Forest | 0.941 | 0.831 |
| Logistic Regression | 0.882 | 0.761 |

### Fairness (IBM AIF360) — Teen vs Adult Mothers
| Metric | Before | After Reweighing | Threshold |
|--------|--------|-----------------|-----------|
| Disparate Impact | measured | mitigated | ≥ 0.80 |
| Statistical Parity Diff. | measured | mitigated | ± 0.10 |
| Equal Opportunity Diff. | measured | mitigated | ± 0.10 |
| Average Odds Diff. | measured | mitigated | ± 0.10 |

### Security (IBM ART)
| Attack | Robustness |
|--------|-----------|
| Gaussian Noise ±5% | ~80% |
| Iterative Black-Box | ~30% (shows need for input validation) |
| Most exploitable vital | Blood Glucose (BS) |

---

## Setup

```bash
git clone https://github.com/RumaizaNorova/MaternaAI.git
cd MaternaAI

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: IBM Granite / LLM explanations
echo "HF_TOKEN=your_token" > .env

streamlit run app.py --server.port 8502
```

### Requirements
- Python 3.12 (3.14 breaks aif360/numpy)
- `aif360` — IBM AI Fairness 360
- `adversarial-robustness-toolbox` — IBM ART
- `scikit-learn >= 1.4.0`
- `streamlit >= 1.32.0`
- `plotly`, `pandas`, `numpy`, `python-dotenv`, `huggingface_hub`

---

## Prize Tracks

| Track | Why MaternaAI Qualifies |
|-------|------------------------|
| **Best Use of IBM Tech** | 5 IBM tools — AIF360, ART, UQ360, Granite, watsonx.governance |
| **Healthcare Track** | WHO dataset · WHO guidelines · clinical decision support |
| **Best UN Hack** | SDG 3 (Good Health) + SDG 10 (Reduced Inequalities) |
| **Best Underprivileged Country** | Targets LMICs — Bangladesh, Uganda, rural clinics |
| **Best Women Hack** | Reduces algorithmic bias against teenage mothers |
| **Best Startup Potential** | Deployable API, clear monetization, NGO partnership path |
| **Best Cybersecurity & Trust** | IBM ART adversarial audit + input validation recommendations |

---

## Clinical Impact

> A rural health worker in a low-resource clinic enters a patient's six vitals. In under 3 seconds, MaternaAI returns a risk classification with a 90% confidence interval, SHAP-style feature attribution explaining *why*, and a 3-sentence IBM Granite clinical brief with the single most important immediate action — all calibrated and audited for bias against teenage patients.

This is not a dashboard demo. This is a deployable triage tool that could be integrated into existing community health worker apps (like DHIS2 or CommCare) as a REST API.

---

## File Structure

```
MaternaAI/
├── app.py              # Streamlit dashboard — IBM Carbon Design System
├── data.py             # WHO dataset loading + feature engineering
├── model.py            # Model training + prediction (3 sklearn models)
├── fairness_engine.py  # IBM AIF360 bias audit + Reweighing mitigation
├── uncertainty.py      # IBM UQ360 methodology — bootstrap CI + calibration
├── adversarial.py      # IBM ART security audit + feature sensitivity
├── granite.py          # IBM Granite LLM engine (Qwen fallback via HF)
├── requirements.txt
└── .env                # HF_TOKEN (not committed)
```

---

*Built in 72 hours at IBM Z × UNSA Sheridan Hackathon 2026.*
*Addressing UN SDG 3 (Good Health and Well-Being) and SDG 10 (Reduced Inequalities).*
