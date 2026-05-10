"""
IBM Granite-3.3-8b via HuggingFace Inference API.
Real IBM foundation model — no IBM Cloud account needed.
"""
import os, requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN   = os.getenv("HF_TOKEN", "")
MODEL_ID   = "ibm-granite/granite-3.3-8b-instruct"
API_URL    = f"https://api-inference.huggingface.co/models/{MODEL_ID}/v1/chat/completions"

def _granite(system: str, user: str, max_tokens: int = 280) -> str:
    if not HF_TOKEN:
        return ""
    try:
        r = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json={
                "model": MODEL_ID,
                "messages": [{"role": "system", "content": system},
                             {"role": "user",   "content": user}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=25,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

def explain_risk(vitals: dict, risk_label: str, probs: list, top_factors: list) -> str:
    factors = ", ".join([f.replace("_", " ") for f, _ in top_factors[:3]])
    text = _granite(
        system=(
            "You are a compassionate maternal health AI assistant trained on WHO clinical guidelines. "
            "You speak directly to a rural healthcare worker in plain language. "
            "Never use jargon. Always include one immediate action step."
        ),
        user=(
            f"Patient vitals: Age {vitals.get('Age')}, SBP {vitals.get('SystolicBP')} mmHg, "
            f"DBP {vitals.get('DiastolicBP')} mmHg, Blood glucose {vitals.get('BS')} mmol/L, "
            f"Temp {vitals.get('BodyTemp')}°F, HR {vitals.get('HeartRate')} bpm.\n"
            f"Risk assessment: {risk_label} "
            f"(Low {probs[0]:.0%} · Medium {probs[1]:.0%} · High {probs[2]:.0%})\n"
            f"Key warning factors: {factors}\n\n"
            f"Write 3 sentences: (1) state the risk level clearly, "
            f"(2) explain the main clinical concern in plain words, "
            f"(3) give the most important immediate action for the health worker."
        ),
    )
    if text:
        return text
    # Fallback
    actions = {
        "Low Risk":    "Continue routine antenatal monitoring every 4 weeks.",
        "Medium Risk": "Refer to facility-level care within 24 hours for closer monitoring.",
        "High Risk":   "IMMEDIATE referral to obstetric emergency care. Do not delay.",
    }
    return (
        f"This patient is assessed as **{risk_label}** "
        f"(Low {probs[0]:.0%} · Medium {probs[1]:.0%} · High {probs[2]:.0%}). "
        f"The primary clinical signals are {factors}. "
        f"{actions[risk_label]}"
    )

def governance_policy(orig: dict, mit: dict) -> str:
    di_o = orig.get("disparate_impact", 1.0)
    di_m = mit.get("disparate_impact", di_o) if "error" not in mit else di_o
    text = _granite(
        system="You are an IBM AI governance expert specialising in healthcare equity and WHO maternal health policy.",
        user=(
            f"IBM AIF360 audit — teenage mothers (≤19) vs adult mothers (≥20):\n"
            f"Disparate Impact before mitigation: {di_o:.3f} (fair ≥ 0.80)\n"
            f"Disparate Impact after Reweighing: {di_m:.3f}\n"
            f"Status: {'FAIR' if di_o >= 0.8 else 'BIASED — mitigation applied'}\n\n"
            f"Give 2 specific policy recommendations referencing IBM AIF360 and WHO guidelines. 2 sentences max."
        ),
        max_tokens=160,
    )
    if text:
        return text
    if di_o >= 0.8:
        return (
            f"Model passes IBM AIF360 fairness standards (DI={di_o:.3f}). "
            "Recommend quarterly bias re-audits as new patient cohorts are added, "
            "per WHO Digital Health Guidelines on algorithmic accountability in LMIC settings."
        )
    return (
        f"Bias detected against teenage mothers (DI={di_o:.3f} → {di_m:.3f} after Reweighing). "
        "Deploy IBM AIF360 Reweighing in production and implement equalized opportunity thresholds "
        "to ensure teenage mothers receive equivalent high-risk detection rates as adult patients."
    )

def is_live() -> bool:
    return bool(HF_TOKEN)
