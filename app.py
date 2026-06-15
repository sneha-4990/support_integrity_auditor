import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
import re
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────────────────
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

PRIORITY_MAP     = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
PRIORITY_MAP_INV = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

# ── Load model (cached so it only loads once) ─────────────────────────────────

@st.cache_resource
def load_model():
    # Load from HuggingFace — replace with your username
    tokenizer = AutoTokenizer.from_pretrained("doingshit/Support_integrity_auditor")
    model     = AutoModelForSequenceClassification.from_pretrained("doingshit/Support_integrity_auditor")
    model.eval()
    return tokenizer, model



# ── Helper functions ──────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def rule_based_severity(text):
    score = 0.0
    words = text.split()
    negated = set()
    neg_words = ["not","no","never","cannot","can't","won't","doesn't","didn't"]
    for i, w in enumerate(words):
        if w in neg_words:
            for j in range(i+1, min(i+4, len(words))):
                negated.add(j)
    for kw in CRITICAL_KEYWORDS:
        if kw in text:
            pos = len(text[:text.find(kw)].split())
            if pos not in negated:
                score += 0.3
    for kw in HIGH_KEYWORDS:
        if kw in text:
            pos = len(text[:text.find(kw)].split())
            if pos not in negated:
                score += 0.15
    for kw in LOW_KEYWORDS:
        if kw in text:
            score -= 0.2
    score += text.count("!") * 0.05
    return float(np.clip(score, 0.0, 1.0))

def infer_severity(rule_score, resolution_hours, satisfaction_score):
    res_norm  = min(resolution_hours / 120.0, 1.0)
    dis_score = (5 - satisfaction_score) / 4.0
    fused = 0.50 * rule_score + 0.30 * res_norm + 0.20 * dis_score
    if fused >= 0.75:   return 3
    elif fused >= 0.55: return 2
    elif fused >= 0.35: return 1
    else:               return 0

def extract_keyword_evidence(text):
    evidence = []
    tl = text.lower()
    for kw in CRITICAL_KEYWORDS:
        if kw in tl:
            evidence.append({"signal":"keyword","value":kw,"weight":"HIGH — critical urgency indicator"})
    for kw in HIGH_KEYWORDS:
        if kw in tl:
            evidence.append({"signal":"keyword","value":kw,"weight":"MEDIUM — elevated urgency indicator"})
    for kw in LOW_KEYWORDS:
        if kw in tl:
            evidence.append({"signal":"keyword","value":kw,"weight":"LOW — routine inquiry indicator"})
    return evidence[:4]

def get_resolution_evidence(hours, priority):
    expected = {"Low":12,"Medium":24,"High":48,"Critical":6}
    exp = expected.get(priority, 24)
    if hours > exp * 2:
        interp = f"Took {hours}h vs expected {exp}h — suggests higher true severity"
    elif hours < exp * 0.3:
        interp = f"Resolved in {hours}h vs expected {exp}h — suggests lower true severity"
    else:
        interp = f"Resolved in {hours}h — consistent with {priority} expectation of {exp}h"
    return {"signal":"resolution_time","value":f"{hours} hours","interpretation":interp}

def generate_dossier_single(ticket_id, subject, description, priority,
                             channel, category, resolution_hours, satisfaction):
    full_text    = clean_text(subject) + " " + clean_text(description)
    rule_score   = rule_based_severity(full_text)
    inferred_num = infer_severity(rule_score, resolution_hours, satisfaction)
    assigned_num = PRIORITY_MAP.get(priority, 1)
    inferred_lbl = PRIORITY_MAP_INV[inferred_num]
    delta        = inferred_num - assigned_num

    if abs(delta) < 1 or (assigned_num <= 1 and abs(delta) < 2):
        mtype = "Consistent"
    elif delta > 0:
        mtype = "Hidden Crisis"
    else:
        mtype = "False Alarm"

    delta_map = {
        -3:"Critical→Low",-2:"two levels over-prioritized",
        -1:"one level over-prioritized",0:"consistent",
        1:"one level under-prioritized",2:"two levels under-prioritized",
        3:"three levels under-prioritized"
    }

    evidence = extract_keyword_evidence(full_text)
    evidence.append(get_resolution_evidence(resolution_hours, priority))
    evidence.append({
        "signal":"satisfaction_score",
        "value":f"{satisfaction}/5",
        "interpretation": (
            "Very low — likely mishandled" if satisfaction<=2
            else "High — handled adequately" if satisfaction>=4
            else "Neutral experience"
        )
    })

    if mtype == "Hidden Crisis":
        analysis = (
            f"Ticket via {channel} under '{category}' was assigned {priority}, "
            f"but urgency signals in '{subject}' indicate higher severity. "
            f"Resolution: {resolution_hours}h, Satisfaction: {satisfaction}/5 — inferred: {inferred_lbl}."
        )
    elif mtype == "False Alarm":
        analysis = (
            f"Ticket via {channel} under '{category}' was assigned {priority}, "
            f"but text shows routine inquiry language with no critical markers. "
            f"Resolution: {resolution_hours}h, Satisfaction: {satisfaction}/5 — inferred: {inferred_lbl}."
        )
    else:
        analysis = (
            f"Ticket via {channel} under '{category}' with assigned {priority} "
            f"aligns with inferred severity {inferred_lbl}. No significant mismatch detected."
        )

    return {
        "ticket_id":          str(ticket_id),
        "assigned_priority":  priority,
        "inferred_severity":  inferred_lbl,
        "mismatch_type":      mtype,
        "severity_delta":     delta_map.get(delta, f"delta={delta}"),
        "feature_evidence":   evidence,
        "constraint_analysis": analysis,
        "confidence":         f"{min(0.95 + rule_score * 0.05, 0.9999):.2%}"
    }

def predict_single(tokenizer, model, text):
    enc = tokenizer(
        text, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        out   = model(**enc)
        probs = torch.softmax(out.logits.float(), dim=-1)[0]
    return int(probs.argmax()), float(probs.max())

# ── Load pre-computed data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Load from HuggingFace dataset files
    from huggingface_hub import hf_hub_download
    import json

    df = pd.read_csv(
        hf_hub_download(
            repo_id="sneha-4990/support-integrity-auditor",
            filename="tickets_with_predictions.csv",
            repo_type="model"
        )
    )
    dossiers = json.load(open(
        hf_hub_download(
            repo_id="sneha-4990/support-integrity-auditor",
            filename="dossiers.json",
            repo_type="model"
        ), encoding="utf-8"
    ))
    summary = json.load(open(
        hf_hub_download(
            repo_id="sneha-4990/support-integrity-auditor",
            filename="summary.json",
            repo_type="model"
        )
    ))
    return df, dossiers, summary

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔍 Support Integrity Auditor (SIA)")
st.markdown("*Semantics-driven priority mismatch detection for CRM support tickets*")

tokenizer, model_loaded = load_model()
df, dossiers, summary   = load_data()

tabs = st.tabs(["📊 Dashboard", "🎫 Single Ticket", "📁 Batch CSV", "📋 Dossier Explorer"])

# ── TAB 1: DASHBOARD ──────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Priority Mismatch Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets",    f"{summary['total_tickets']:,}")
    col2.metric("Mismatches Found", f"{summary['total_mismatches']:,}",
                f"{summary['total_mismatches']/summary['total_tickets']:.1%}")
    col3.metric("Hidden Crises",    f"{summary['hidden_crisis']:,}")
    col4.metric("False Alarms",     f"{summary['false_alarm']:,}")

    st.divider()
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Mismatch Type Distribution")
        fig_pie = px.pie(
            values=[summary["hidden_crisis"], summary["false_alarm"], summary["consistent"]],
            names=["Hidden Crisis", "False Alarm", "Consistent"],
            color_discrete_map={"Hidden Crisis":"#e74c3c","False Alarm":"#f39c12","Consistent":"#2ecc71"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col6:
        st.subheader("Severity Delta Heatmap")
        heatmap_data = df.groupby(["Issue_Category","Priority_Level"])["severity_delta"].mean().unstack(fill_value=0)
        fig_heat = px.imshow(
            heatmap_data, color_continuous_scale="RdYlGn_r",
            title="Avg Severity Delta by Category & Priority"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Mismatches by Channel")
    channel_data = df[df["pred_label"]==1]["Ticket_Channel"].value_counts().reset_index()
    channel_data.columns = ["Channel","Count"]
    fig_bar = px.bar(channel_data, x="Channel", y="Count",
                     color="Channel", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Model Performance")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Test Accuracy", f"{summary['test_accuracy']:.2%}")
    mc2.metric("Macro F1",      f"{summary['test_f1']:.4f}")
    mc3.metric("Recall (Consistent)", f"{summary['recall_consistent']:.2%}")
    mc4.metric("Recall (Mismatch)",   f"{summary['recall_mismatch']:.2%}")

# ── TAB 2: SINGLE TICKET ─────────────────────────────────────────────────────
with tabs[1]:
    st.header("Analyze a Single Ticket")

    c1, c2 = st.columns(2)
    with c1:
        t_id      = st.text_input("Ticket ID", value="TKT-NEW-001")
        t_subject = st.text_input("Subject", value="Cannot access my account")
        t_desc    = st.text_area("Description",
            value="I have been locked out of my account for 3 days. "
                  "I cannot access any of my data and this is extremely urgent. "
                  "Please escalate immediately.", height=120)
    with c2:
        t_priority = st.selectbox("Assigned Priority", ["Low","Medium","High","Critical"])
        t_channel  = st.selectbox("Channel", ["Chat","Email","Web Form"])
        t_category = st.selectbox("Category", ["Technical","Billing","Account","General Inquiry","Fraud"])
        t_hours    = st.slider("Resolution Time (hours)", 1, 120, 24)
        t_sat      = st.slider("Satisfaction Score", 1, 5, 3)

    if st.button("🔍 Analyze Ticket", type="primary"):
        model_input = (
            f"Subject: {t_subject} [SEP] Description: {t_desc} "
            f"[SEP] Channel: {t_channel} [SEP] Category: {t_category} "
            f"[SEP] Resolution: {t_hours} hours [SEP] Priority: {t_priority}"
        )
        pred_label, confidence = predict_single(tokenizer, model_loaded, model_input)
        dossier = generate_dossier_single(
            t_id, t_subject, t_desc, t_priority,
            t_channel, t_category, t_hours, t_sat
        )

        if pred_label == 1:
            mtype = dossier["mismatch_type"]
            color = "#e74c3c" if mtype == "Hidden Crisis" else "#f39c12"
            st.markdown(f"""
            <div style='background:{color};padding:15px;border-radius:10px;color:white'>
            <h3>⚠️ MISMATCH DETECTED — {mtype}</h3>
            <b>Assigned:</b> {dossier['assigned_priority']} &nbsp;|&nbsp;
            <b>Inferred:</b> {dossier['inferred_severity']} &nbsp;|&nbsp;
            <b>Confidence:</b> {dossier['confidence']}
            </div>""", unsafe_allow_html=True)
        else:
            st.success(f"✅ **CONSISTENT** — Priority appears correctly assigned. Confidence: {confidence:.2%}")

        st.subheader("📋 Evidence Dossier")
        st.json(dossier)

# ── TAB 3: BATCH CSV ─────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Batch CSV Upload")
    st.markdown("Upload a CSV with columns: `Ticket_ID, Ticket_Subject, Ticket_Description, Priority_Level, Ticket_Channel, Issue_Category, Resolution_Time_Hours, Satisfaction_Score`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df)} tickets")

        if st.button("🔍 Run Batch Analysis", type="primary"):
            results = []
            progress = st.progress(0)
            for i, row in batch_df.iterrows():
                model_input = (
                    f"Subject: {row.get('Ticket_Subject','')} [SEP] "
                    f"Description: {row.get('Ticket_Description','')} [SEP] "
                    f"Channel: {row.get('Ticket_Channel','')} [SEP] "
                    f"Category: {row.get('Issue_Category','')} [SEP] "
                    f"Resolution: {row.get('Resolution_Time_Hours',24)} hours [SEP] "
                    f"Priority: {row.get('Priority_Level','Medium')}"
                )
                pred, conf = predict_single(tokenizer, model_loaded, model_input)
                dossier = generate_dossier_single(
                    row.get("Ticket_ID", i),
                    row.get("Ticket_Subject",""),
                    row.get("Ticket_Description",""),
                    row.get("Priority_Level","Medium"),
                    row.get("Ticket_Channel","Email"),
                    row.get("Issue_Category","General Inquiry"),
                    row.get("Resolution_Time_Hours", 24),
                    row.get("Satisfaction_Score", 3)
                )
                results.append({
                    "Ticket_ID":       row.get("Ticket_ID", i),
                    "Assigned":        row.get("Priority_Level",""),
                    "Inferred":        dossier["inferred_severity"],
                    "Prediction":      "Mismatch" if pred==1 else "Consistent",
                    "Mismatch_Type":   dossier["mismatch_type"],
                    "Confidence":      f"{conf:.2%}",
                    "Severity_Delta":  dossier["severity_delta"]
                })
                progress.progress((i+1)/len(batch_df))

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            csv_out = results_df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv_out,
                               "sia_results.csv", "text/csv")

# ── TAB 4: DOSSIER EXPLORER ───────────────────────────────────────────────────
with tabs[3]:
    st.header("Dossier Explorer")
    st.markdown(f"Browsing **{len(dossiers)}** flagged ticket dossiers")

    filter_type = st.selectbox("Filter by mismatch type",
                                ["All","Hidden Crisis","False Alarm"])
    filtered = dossiers if filter_type=="All" else [
        d for d in dossiers if d["mismatch_type"]==filter_type
    ]

    st.write(f"Showing {len(filtered)} dossiers")
    idx = st.slider("Select dossier", 0, max(len(filtered)-1,1), 0)
    if filtered:
        st.json(filtered[idx])
