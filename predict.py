"""
predict.py
Support Integrity Auditor (SIA) — Inference Script
Usage: python predict.py --input tickets.csv --output results.csv --model final_model
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

CRITICAL_KEYWORDS = [
    "fraud","unauthorized","hacked","breach","stolen","compromised",
    "cannot access","system down","outage","not working","broken",
    "data loss","account locked","suspended","terminated","urgent",
    "immediately","asap","emergency","critical","severe","blocked",
    "payment failed","charge","refund","dispute","escalate"
]
HIGH_KEYWORDS = [
    "error","issue","problem","fail","unable","wrong","incorrect",
    "missing","lost","delay","slow","not received","pending",
    "frustrated","disappointed","unacceptable","still waiting"
]
LOW_KEYWORDS = [
    "inquiry","question","how to","where is","what is","information",
    "curious","wondering","just wanted","could you","please let me know",
    "general","hours of operation","location"
]

PRIORITY_MAP     = {"Low":0,"Medium":1,"High":2,"Critical":3}
PRIORITY_MAP_INV = {0:"Low",1:"Medium",2:"High",3:"Critical"}

def clean_text(text):
    text = str(text).lower().strip()
    return re.sub(r'\s+', ' ', text)

def rule_based_severity(text):
    score = 0.0
    words = text.split()
    negated = set()
    for i, w in enumerate(words):
        if w in ["not","no","never","cannot","can't","won't"]:
            for j in range(i+1, min(i+4, len(words))):
                negated.add(j)
    for kw in CRITICAL_KEYWORDS:
        if kw in text:
            pos = len(text[:text.find(kw)].split())
            if pos not in negated: score += 0.3
    for kw in HIGH_KEYWORDS:
        if kw in text:
            pos = len(text[:text.find(kw)].split())
            if pos not in negated: score += 0.15
    for kw in LOW_KEYWORDS:
        if kw in text: score -= 0.2
    return float(np.clip(score + text.count("!")*0.05, 0.0, 1.0))

def infer_severity(rule_score, resolution_hours, satisfaction):
    fused = (0.50*rule_score +
             0.30*min(resolution_hours/120.0,1.0) +
             0.20*((5-satisfaction)/4.0))
    if fused>=0.75:   return 3
    elif fused>=0.55: return 2
    elif fused>=0.35: return 1
    else:             return 0

def generate_dossier(row, pred_label, confidence):
    subject     = str(row.get("Ticket_Subject",""))
    description = str(row.get("Ticket_Description",""))
    priority    = str(row.get("Priority_Level","Medium"))
    channel     = str(row.get("Ticket_Channel","Email"))
    category    = str(row.get("Issue_Category","General"))
    hours       = float(row.get("Resolution_Time_Hours", 24))
    satisfaction= float(row.get("Satisfaction_Score", 3))
    ticket_id   = str(row.get("Ticket_ID", "N/A"))

    full_text    = clean_text(subject) + " " + clean_text(description)
    rule_score   = rule_based_severity(full_text)
    inferred_num = infer_severity(rule_score, hours, satisfaction)
    assigned_num = PRIORITY_MAP.get(priority, 1)
    inferred_lbl = PRIORITY_MAP_INV[inferred_num]
    delta        = inferred_num - assigned_num

    if pred_label == 0:
        mtype = "Consistent"
    elif delta > 0:
        mtype = "Hidden Crisis"
    else:
        mtype = "False Alarm"

    # Feature evidence — all grounded in ticket fields
    evidence = []
    for kw in CRITICAL_KEYWORDS:
        if kw in full_text:
            evidence.append({"signal":"keyword","value":kw,"weight":"HIGH — critical urgency"})
    for kw in HIGH_KEYWORDS:
        if kw in full_text:
            evidence.append({"signal":"keyword","value":kw,"weight":"MEDIUM — elevated urgency"})
    for kw in LOW_KEYWORDS:
        if kw in full_text:
            evidence.append({"signal":"keyword","value":kw,"weight":"LOW — routine inquiry"})
    evidence = evidence[:4]

    expected = {"Low":12,"Medium":24,"High":48,"Critical":6}
    exp_h    = expected.get(priority, 24)
    res_interp = (f"Took {hours}h vs expected {exp_h}h — higher severity likely"
                  if hours > exp_h*2 else
                  f"Resolved in {hours}h — consistent with {priority}")
    evidence.append({"signal":"resolution_time","value":f"{hours}h","interpretation":res_interp})
    evidence.append({"signal":"satisfaction_score","value":f"{satisfaction}/5",
                     "interpretation":"Low — likely mishandled" if satisfaction<=2
                     else "High — handled adequately" if satisfaction>=4
                     else "Neutral"})

    delta_map = {-3:"Critical→Low",-2:"two levels over-prioritized",
                 -1:"one level over-prioritized",0:"consistent",
                 1:"one level under-prioritized",2:"two levels under-prioritized",
                 3:"three levels under-prioritized"}

    if mtype=="Hidden Crisis":
        analysis = (f"Ticket via {channel} under '{category}' assigned {priority}, "
                    f"but urgency signals in '{subject}' indicate higher severity. "
                    f"Resolution: {hours}h, Satisfaction: {satisfaction}/5 — inferred: {inferred_lbl}.")
    elif mtype=="False Alarm":
        analysis = (f"Ticket via {channel} under '{category}' assigned {priority}, "
                    f"but text shows routine language with no critical markers. "
                    f"Resolution: {hours}h, Satisfaction: {satisfaction}/5 — inferred: {inferred_lbl}.")
    else:
        analysis = (f"Ticket via {channel} under '{category}' with {priority} priority "
                    f"aligns with inferred severity {inferred_lbl}. No mismatch detected.")

    return {
        "ticket_id":          ticket_id,
        "assigned_priority":  priority,
        "inferred_severity":  inferred_lbl,
        "mismatch_type":      mtype,
        "severity_delta":     delta_map.get(delta, f"delta={delta}"),
        "feature_evidence":   evidence,
        "constraint_analysis": analysis,
        "confidence":         f"{confidence:.2%}"
    }

def predict(input_path, output_path, model_path, dossier_path):
    print(f"Loading model from {model_path}...")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model     = model.float().to(device)
    model.eval()

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    def build_input(row):
        return (f"Subject: {row.get('Ticket_Subject','')} "
                f"[SEP] Description: {row.get('Ticket_Description','')} "
                f"[SEP] Channel: {row.get('Ticket_Channel','')} "
                f"[SEP] Category: {row.get('Issue_Category','')} "
                f"[SEP] Resolution: {row.get('Resolution_Time_Hours',24)} hours "
                f"[SEP] Priority: {row.get('Priority_Level','Medium')}")

    results, dossiers = [], []
    print("Running inference...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        model_input = build_input(row)
        enc = tokenizer(
            model_input, max_length=256, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            out   = model(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device))
            probs = torch.softmax(out.logits.float(), dim=-1)[0]

        pred_label = int(probs.argmax())
        confidence = float(probs.max())

        results.append({
            "Ticket_ID":     row.get("Ticket_ID", i),
            "Assigned":      row.get("Priority_Level",""),
            "Prediction":    "Mismatch" if pred_label==1 else "Consistent",
            "Confidence":    f"{confidence:.2%}"
        })

        if pred_label == 1:
            dossiers.append(generate_dossier(row, pred_label, confidence))

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

    with open(dossier_path, "w", encoding="utf-8") as f:
        json.dump(dossiers, f, indent=2, ensure_ascii=False)
    print(f"✅ {len(dossiers)} dossiers saved to {dossier_path}")

    total     = len(results_df)
    mismatches = (results_df["Prediction"]=="Mismatch").sum()
    print(f"\nSummary: {mismatches}/{total} tickets flagged ({mismatches/total:.1%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Inference Script")
    parser.add_argument("--input",    type=str, required=True,  help="Input CSV path")
    parser.add_argument("--output",   type=str, default="predictions.csv")
    parser.add_argument("--model",    type=str, default="final_model")
    parser.add_argument("--dossiers", type=str, default="dossiers.json")
    args = parser.parse_args()
    predict(args.input, args.output, args.model, args.dossiers)
