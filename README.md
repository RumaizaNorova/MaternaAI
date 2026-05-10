# Live at: https://treading-bottling-dilute.ngrok-free.dev

# MaternaAI

**AI-powered maternal health risk stratification for underserved communities**

*IBM Z × UNSA Sheridan Hackathon 2026 — UN SDG 3 · SDG 10*

---

## The Problem

**287,000 mothers die every year from preventable pregnancy complications. 95% in low-income countries. Teenage mothers face 3× higher mortality — yet are routinely misclassified or ignored by biased clinical AI systems.**

A healthcare worker in rural Bangladesh or Uganda sees dozens of patients with no specialist support. They need a decision tool that is:
- **Accurate** — catches high-risk mothers before a crisis
- **Fair** — doesn't systematically miss teenage patients
- **Explainable** — tells clinicians *why* a patient is high risk and what to change
- **Secure** — resilient to data manipulation at the point of entry
- **Privacy-preserving** — protects patient data under differential privacy

MaternaAI addresses all five.

---

## IBM Tools Integrated (4)

| Tool | How We Use It |
|------|--------------|
| **IBM AI Fairness 360** | Three-strategy bias pipeline: Reweighing (preprocessing) + DisparateImpactRemover (preprocessing) + CalibratedEqOddsPostprocessing (postprocessing). Audits teen vs adult mother disparity across DI, SPD, EOD, AOD, Theil Index. |
| **IBM Adversarial Robustness Toolbox (ART)** | Two attack classes (Gaussian noise + iterative black-box) against high-risk patients. FeatureSqueezing defense (8-bit discretization) applied to each. Attack success rates reported before and after defense. |
| **IBM AIX360** | LIME local explanation (per-patient feature contributions) + counterfactual minimum-change analysis (smallest vital adjustment to drop risk class, with clinical notes). |
| **IBM diffprivlib** | DP-GaussianNB (ε=1.0) binary high-risk detector. BudgetAccountant (ε=10.0 total) tracks cumulative privacy spend per API query (0.05ε each), surfaced live in the UI. |

---

## Architecture

```
WHO Maternal Health Dataset (UCI, 1,014 patients)
        │
        ▼
┌─────────────────────┐
│   data.py           │  Feature engineering, age_group encoding
│   6 vitals          │  Age · SystolicBP · DiastolicBP · BS · BodyTemp · HeartRate
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────────────┐
│   model.py          │     │   IBM AI Fairness 360                 │
│   GBM · RF · LR     │────▶│   3-strategy bias mitigation pipeline │
│   Best: GBM 0.951   │     │   Reweighing + DIR + CalibratedEqOdds│
│   + DP-GaussianNB   │     │   DI · SPD · EOD · AOD · Theil Index │
└────────┬────────────┘     └──────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────────────┐
│   uncertainty.py    │     │   IBM ART                             │
│   CalibratedCV      │────▶│   Gaussian noise attack               │
│   Bootstrap 90% CI  │     │   Iterative black-box attack          │
└────────┬────────────┘     │   FeatureSqueezing defense (8-bit)    │
         │                  └──────────────────────────────────────┘
         ▼
┌─────────────────────┐     ┌──────────────────────────────────────┐
│   explainability.py │     │   IBM diffprivlib                     │
│   IBM AIX360 LIME   │     │   BudgetAccountant ε=10.0             │
│   Counterfactuals   │     │   0.05ε per query, tracked live       │
└────────┬────────────┘     └──────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   server.py (Flask) │  REST API + single-page UI
│   /api/predict      │  Risk · LIME · Counterfactual · ε-budget
│   /api/fairness     │  3-strategy AIF360 results
│   /api/security     │  Attack/defense comparison
│   /api/privacy-budget│ Live differential privacy spend
└─────────────────────┘
```

---

## Model Performance

| Model | AUC-ROC (Macro OvR) | F1 (Macro) |
|-------|--------------------|----|
| **Gradient Boosting** *(selected)* | **0.9513** | **0.843** |
| Random Forest | 0.941 | 0.831 |
| Logistic Regression | 0.882 | 0.761 |
| DP-GaussianNB (ε=1.0) | ~0.83 | — |

---

## Fairness (IBM AIF360) — Teen vs Adult Mothers

| Strategy | Disparate Impact | SPD | Result |
|----------|-----------------|-----|--------|
| Original model | 1.298 | measured | Baseline |
| Reweighing | 1.000 | mitigated | ✓ Pass |
| DisparateImpactRemover | 1.000 | mitigated | ✓ Pass |
| CalibratedEqOddsPostprocessing | 0.439 | measured | Threshold adjustment |

---

## Security (IBM ART)

| Attack | Robustness (undefended) | After FeatureSqueezing |
|--------|------------------------|------------------------|
| Gaussian Noise ±5% | 70% | reported |
| Iterative Black-Box | 45% | reported |
| Most exploitable vital | Age / Blood Glucose | — |

---

## Setup

```bash
git clone https://github.com/RumaizaNorova/MaternaAI.git
cd MaternaAI

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: LLM clinical explanations
echo "HF_TOKEN=your_token" > .env

python server.py   # runs on port 5050
```

### Requirements
- Python 3.12 (3.14 breaks aif360/numpy)
- `aif360`, `aix360`, `adversarial-robustness-toolbox`, `diffprivlib`
- `flask`, `scikit-learn >= 1.4.0`, `pandas`, `numpy`
- `BlackBoxAuditing` (for DisparateImpactRemover): `pip install BlackBoxAuditing`

---

## File Structure

```
MaternaAI/
├── server.py           # Flask REST API (6 endpoints)
├── data.py             # WHO dataset loading + feature engineering
├── model.py            # GBM/RF/LR training + DP-GaussianNB (diffprivlib)
├── fairness_engine.py  # IBM AIF360 — 3-strategy bias pipeline
├── adversarial.py      # IBM ART — attack + FeatureSqueezing defense
├── explainability.py   # IBM AIX360 — LIME + counterfactual explain
├── uncertainty.py      # Bootstrap 90% CI + calibration metrics
├── granite.py          # LLM clinical brief engine
├── templates/index.html# Single-page UI (no framework)
├── requirements.txt
├── Procfile
└── .env                # HF_TOKEN (not committed)
```

---

## Prize Tracks

| Track | Why MaternaAI Qualifies |
|-------|------------------------|
| **Best Use of IBM Tech** | 4 IBM tools deeply integrated — AIF360, ART, AIX360, diffprivlib |
| **Healthcare Track** | WHO dataset · WHO guidelines · clinical decision support tool |
| **Best UN Hack** | SDG 3 (Good Health) + SDG 10 (Reduced Inequalities) |
| **Best Underprivileged Country** | Targets LMICs — rural clinics, no specialist access |
| **Best Women Hack** | Reduces algorithmic bias against teenage mothers |
| **Best Startup Potential** | Deployable REST API, DHIS2/CommCare integration path |
| **Best Cybersecurity & Trust** | IBM ART adversarial audit + FeatureSqueezing defense |

---

*Built at IBM Z × UNSA Sheridan Hackathon 2026.*
*Addressing UN SDG 3 (Good Health and Well-Being) and SDG 10 (Reduced Inequalities).*
