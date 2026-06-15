# 🔍 Support Integrity Auditor (SIA)


**A self-supervised AI system that detects priority mismatches in CRM support tickets — no pre-labeled data required.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](YOUR_STREAMLIT_URL)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📖 What is SIA?

In large customer support teams, agents manually assign priority levels (Low / Medium / High / Critical) to incoming tickets. This process is prone to:

- ❌ **Agent fatigue bias** — late-shift agents may under-prioritize issues
- ❌ **Keyword anchoring** — agents focus on surface words, missing true urgency
- ❌ **Customer favoritism** — VIP customers get inflated priorities
- ❌ **SLA violations** — misclassified tickets breach service agreements

**SIA automatically audits every ticket** and flags two types of mismatches:

| Type | Meaning | Example |
|---|---|---|
| 🚨 Hidden Crisis | Ticket assigned LOW but is actually CRITICAL | Account breach labeled as "Low" |
| ⚠️ False Alarm | Ticket assigned CRITICAL but is actually LOW | General inquiry labeled as "Critical" |

---

## 🎯 Key Features

- ✅ **Fully self-supervised** — generates its own training labels, no manual annotation needed
- ✅ **3-signal fusion** — combines NLP keywords, resolution time, and satisfaction scores
- ✅ **Fine-tuned DeBERTa-v3-small** with LoRA adapters for efficient training
- ✅ **Evidence Dossier** — every flagged ticket gets a structured JSON explanation
- ✅ **Zero hallucination** — all evidence is traceable to actual ticket fields
- ✅ **Streamlit Dashboard** — visual analytics + single ticket + batch CSV analysis

---

## 📊 Results

| Metric | Our Score | Required Threshold | Status |
|---|---|---|---|
| Binary Classification Accuracy | **98.57%** | ≥ 83% | ✅ Passed |
| Macro F1 Score | **0.9808** | ≥ 0.82 | ✅ Passed |
| Per-Class Recall (Consistent) | **99.16%** | ≥ 0.78 | ✅ Passed |
| Per-Class Recall (Mismatch) | **96.78%** | ≥ 0.78 | ✅ Passed |

---

## 🗂️ Dataset

**Source:** [Customer Support Tickets — CRM Dataset (Kaggle)](https://kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data)

- 📦 **20,000 tickets**, zero missing values
- 4 priority levels: Low (38.6%), Medium (37.9%), High (17.1%), Critical (6.5%)
- 3 channels: Chat, Email, Web Form
- 5 categories: Technical, Billing, Account, General Inquiry, Fraud

| Column Used | Role in SIA |
|---|---|
| `Ticket_Subject` | Short summary — keyword signal source |
| `Ticket_Description` | Full text — primary NLP signal source |
| `Priority_Level` | Human-assigned label being audited |
| `Ticket_Channel` | Intake channel — metadata feature |
| `Resolution_Time_Hours` | Time to resolve — indirect severity signal |
| `Satisfaction_Score` | Customer rating (1–5) — dissatisfaction proxy |
| `Issue_Category` | Ticket category — metadata feature |

---

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────────┐

│                    RAW CRM TICKETS (20,000)                  │

└─────────────────────────┬───────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────────┐

│            STAGE 1 — PSEUDO-LABEL GENERATION                │

│                  (Self-Supervised)                           │

│                                                             │

│  ┌─────────────────┐ ┌──────────────────┐ ┌─────────────┐  │

│  │  Signal 1       │ │  Signal 2        │ │  Signal 3   │  │

│  │  Rule-based NLP │ │  Resolution Time │ │ Satisfaction│  │

│  │  Weight: 50%    │ │  Weight: 30%     │ │ Weight: 20% │  │

│  └────────┬────────┘ └────────┬─────────┘ └──────┬──────┘  │

│           └──────────────────┼──────────────────┘          │

│                              ▼                              │

│                    Weighted Signal Fusion                    │

│                              │                              │

│                              ▼                              │

│              Binary Mismatch Label (0 or 1)                 │

└─────────────────────────┬───────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────────┐

│            STAGE 2 — FINE-TUNED CLASSIFIER                  │

│                                                             │

│   Model:   DeBERTa-v3-small + LoRA (r=16, alpha=32)        │

│   Input:   Text fields + Metadata (channel, category,       │

│            resolution time, priority)                       │

│   Handles: Class imbalance via weighted CrossEntropyLoss    │

│   Trained: 3 epochs, AdamW + linear warmup scheduler        │

└─────────────────────────┬───────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────────┐

│            STAGE 3 — EVIDENCE DOSSIER GENERATION            │

│                                                             │

│   For every flagged ticket → structured JSON with:          │

│   • Assigned priority vs Inferred severity                  │

│   • Mismatch type (Hidden Crisis / False Alarm)             │

│   • Feature evidence (grounded in ticket fields only)       │

│   • 2-3 sentence constraint analysis                        │

│   • Confidence score                                        │

└─────────────────────────────────────────────────────────────┘

---

## 🔬 Methodology Deep Dive

### Stage 1 — Pseudo-Label Generation

Since there are no pre-annotated mismatch labels, SIA bootstraps its own supervision signal using **3 independent signals**:

#### Signal 1 — Rule-based NLP (Weight: 50%)
Scans ticket text for urgency keywords with negation detection:
- **Critical keywords** (+0.30 each): `fraud`, `unauthorized`, `account locked`, `system down`, `data loss`, `escalate`, etc.
- **High keywords** (+0.15 each): `error`, `issue`, `unable`, `frustrated`, `unacceptable`, etc.
- **Low keywords** (−0.20 each): `inquiry`, `how to`, `where is`, `hours of operation`, etc.
- **Negation detection**: Words like `not`, `never`, `cannot` within 3 words of a keyword reduce its score

#### Signal 2 — Resolution Time Regression (Weight: 30%)
- Trains a `LinearRegression` to predict resolution time from channel + category + satisfaction
- Computes **residual** = actual time − predicted time
- High positive residual = ticket took much longer than expected = higher true severity

#### Signal 3 — Satisfaction Score (Weight: 20%)
- Low satisfaction (1–2) on a Low-priority ticket = likely mishandled
- Inverted and normalized: `dissatisfaction = (5 − score) / 4`

#### Signal Fusion Strategy
```python
inferred_severity = (0.50 × signal_rule +
                     0.30 × signal_resolution +
                     0.20 × signal_satisfaction)
```

Mismatch label logic:
- For Low/Medium assigned tickets: flag if delta ≥ 2 levels (avoids noisy borderline cases)
- For High/Critical assigned tickets: flag if delta ≥ 1 level

**Result:** 24.8% mismatch rate (4,968 mismatches out of 20,000 tickets)

---

### Stage 1 Ablation — Signal Contribution

| Signal Combination | Mismatch Rate | Notes |
|---|---|---|
| Signal 1 only (Rule NLP) | ~18% | Text-only, misses metadata |
| Signal 1 + 2 (Rule + Resolution) | ~22% | Better but ignores satisfaction |
| All 3 signals (Full fusion) | **24.8%** | Best balance — used for training |

### Signal Agreement (Cohen's Kappa)

| Signal Pair | κ Score | Interpretation |
|---|---|---|
| Rule NLP vs Resolution | −0.006 | Nearly independent ✅ |
| Rule NLP vs Satisfaction | 0.058 | Nearly independent ✅ |
| Resolution vs Satisfaction | −0.004 | Nearly independent ✅ |

> Low kappa values confirm all 3 signals measure **independent aspects** of severity, justifying multi-signal fusion over single-signal approaches.

---

### Stage 2 — Classifier Training Details

| Component | Choice | Reason |
|---|---|---|
| Base model | DeBERTa-v3-small | Best-in-class for text classification, small enough for free GPU |
| Adapter | LoRA (r=16, alpha=32) | Reduces trainable params from 141M to ~2M, prevents overfitting |
| Input format | Text + metadata as single string | Allows model to attend across all features |
| Loss function | Weighted CrossEntropy | Handles 3:1 class imbalance (Consistent vs Mismatch) |
| Optimizer | AdamW (lr=2e-4) | Standard for transformer fine-tuning |
| Scheduler | Linear warmup (10%) + decay | Prevents early overshooting |
| Epochs | 3 | Sufficient — model converges by epoch 2 |

---

### Stage 3 — Evidence Dossier Schema

```json
{
  "ticket_id": "TKT-100000",
  "assigned_priority": "High",
  "inferred_severity": "Low",
  "mismatch_type": "False Alarm",
  "severity_delta": "two levels over-prioritized",
  "feature_evidence": [
    {
      "signal": "keyword",
      "value": "where is",
      "weight": "LOW — routine inquiry indicator"
    },
    {
      "signal": "keyword",
      "value": "hours of operation",
      "weight": "LOW — routine inquiry indicator"
    },
    {
      "signal": "resolution_time",
      "value": "43 hours",
      "interpretation": "Resolved in 43h — consistent with High expectation of 48h"
    },
    {
      "signal": "satisfaction_score",
      "value": "5/5",
      "interpretation": "High — suggests issue was handled adequately"
    }
  ],
  "constraint_analysis": "Ticket assigned High priority via Web Form under 'General Inquiry', but the description contains primarily routine inquiry language with no critical urgency markers. Resolution completed in 43 hours with satisfaction 5/5, consistent with a lower severity — inferred severity: Low.",
  "confidence": "99.98%"
}
```

> ⚠️ **Hard Rule:** Every `feature_evidence` item is traceable to a specific field in the input ticket. Fabricated or unverifiable claims result in immediate disqualification.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA GPU recommended (works on CPU but slower)

### Installation

```bash
# Clone the repository
git clone https://github.com/sneha-4990/support_integrity_auditor.git
cd support_integrity_auditor

# Install dependencies
pip install -r requirements.txt
```

### Train from Scratch

```bash
python train_pipeline.py \
  --data customer_support_tickets.csv \
  --output_dir final_model \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-4
```

### Run Inference on New Tickets

```bash
python predict.py \
  --input new_tickets.csv \
  --output predictions.csv \
  --model final_model \
  --dossiers dossiers.json
```

Input CSV must have these columns:
`Ticket_ID, Ticket_Subject, Ticket_Description, Priority_Level, Ticket_Channel, Issue_Category, Resolution_Time_Hours, Satisfaction_Score`

### Launch Streamlit App Locally

```bash
streamlit run app.py
```

---

## 📁 Repository Structure
support_integrity_auditor/

│

├── app.py                  # Streamlit web application

│   ├── Dashboard tab       # Mismatch analytics + heatmaps

│   ├── Single Ticket tab   # Analyze one ticket with dossier

│   ├── Batch CSV tab       # Upload and analyze multiple tickets

│   └── Dossier Explorer    # Browse all flagged ticket dossiers

│

├── train_pipeline.py       # Complete standalone training script

├── predict.py              # Inference script → predictions + dossiers

├── notebook.ipynb          # Full reproducible Colab pipeline

├── requirements.txt        # Pinned Python dependencies

└── README.md               # This file

---

## 🖥️ Streamlit Web App

**Live URL:** supportintegrityauditor.streamlit.app

The app has 4 tabs:

| Tab | Description |
|---|---|
| 📊 Dashboard | Mismatch distribution, severity delta heatmap, channel breakdown, model metrics |
| 🎫 Single Ticket | Input any ticket manually and get instant judgment + full dossier |
| 📁 Batch CSV | Upload a CSV of tickets, get predictions + downloadable results |
| 📋 Dossier Explorer | Browse all 4,906 flagged ticket dossiers with filtering |

---

## 📹 Demo Video

[Link to demo video]

The demo covers:
- Walkthrough of one **Hidden Crisis** case
- Walkthrough of one **False Alarm** case
- Explanation of the pseudo-label generation strategy
- Live adversarial ticket input demonstration

---

## ✅ Deliverables Checklist

- [x] `notebook.ipynb` — Full reproducible pipeline
- [x] `train_pipeline.py` — Standalone training script
- [x] `predict.py` — Inference script with dossier output
- [x] `README.md` — Methodology, architecture, ablation, metrics
- [x] `requirements.txt` — Pinned dependencies
- [x] Streamlit Web App — Hosted public URL
- [x] Demo Video — ~3 minutes

---

## 🏆 Evaluation Summary

| Criteria | Result |
|---|---|
| Binary Accuracy ≥ 83% | ✅ 98.57% |
| Macro F1 ≥ 0.82 | ✅ 0.9808 |
| Per-Class Recall ≥ 0.78 | ✅ 99.16% / 96.78% |
| Zero hallucination in dossiers | ✅ All evidence grounded |
| Self-supervised pseudo-labeling | ✅ 3-signal fusion |
| Fine-tuned model (not frozen) | ✅ DeBERTa + LoRA |

---

*Built for MARS Open Projects 2026 — Models and Robotics Section*
*Artificial Intelligence / Machine Learning — NLP & CRM Systems*


