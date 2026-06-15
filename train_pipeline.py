"""
train_pipeline.py
Support Integrity Auditor (SIA) — Standalone Training Script
Usage: python train_pipeline.py --data customer_support_tickets.csv
"""

import argparse
import os
import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, cohen_kappa_score
)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType

# ── Keyword lists ──────────────────────────────────────────────────────────────
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

# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\!\?\,\-]', '', text)
    return text

# ── Signal 1: Rule-based NLP ───────────────────────────────────────────────────
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

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(df):
    df["clean_subject"]     = df["Ticket_Subject"].apply(clean_text)
    df["clean_description"] = df["Ticket_Description"].apply(clean_text)
    df["full_text"]         = df["clean_subject"] + " [SEP] " + df["clean_description"]

    priority_map    = {"Low":0,"Medium":1,"High":2,"Critical":3}
    df["priority_num"] = df["Priority_Level"].map(priority_map)

    scaler = MinMaxScaler()
    df["resolution_norm"] = scaler.fit_transform(df[["Resolution_Time_Hours"]])
    df["dissatisfaction"]  = (5 - df["Satisfaction_Score"]) / 4.0

    le_channel  = LabelEncoder()
    le_category = LabelEncoder()
    df["channel_enc"]  = le_channel.fit_transform(df["Ticket_Channel"])
    df["category_enc"] = le_category.fit_transform(df["Issue_Category"])
    return df

# ── Stage 1: Pseudo-label generation ──────────────────────────────────────────
def generate_pseudo_labels(df):
    # Signal 1: Rule NLP
    df["signal_rule"] = df["full_text"].apply(rule_based_severity)

    # Signal 2: Resolution-time regression
    reg_features = df[["channel_enc","category_enc","Satisfaction_Score"]].values
    reg_target   = df["Resolution_Time_Hours"].values
    reg_model    = LinearRegression()
    reg_model.fit(reg_features, reg_target)
    df["predicted_resolution"] = reg_model.predict(reg_features)
    df["resolution_residual"]  = df["Resolution_Time_Hours"] - df["predicted_resolution"]
    res_scaler = MinMaxScaler()
    df["signal_resolution"] = res_scaler.fit_transform(
        df["resolution_residual"].values.reshape(-1,1)
    ).flatten()

    # Signal 3: Satisfaction
    df["signal_satisfaction"] = df["dissatisfaction"]

    # Fusion
    df["inferred_severity_score"] = (
        0.50 * df["signal_rule"] +
        0.30 * df["signal_resolution"] +
        0.20 * df["signal_satisfaction"]
    )

    def score_to_severity(s):
        if s >= 0.75:   return 3
        elif s >= 0.55: return 2
        elif s >= 0.35: return 1
        else:           return 0

    df["inferred_severity_num"] = df["inferred_severity_score"].apply(score_to_severity)
    inv_map = {0:"Low",1:"Medium",2:"High",3:"Critical"}
    df["inferred_severity"] = df["inferred_severity_num"].map(inv_map)
    df["severity_delta"]    = df["inferred_severity_num"] - df["priority_num"]

    def compute_mismatch(row):
        delta    = abs(row["inferred_severity_num"] - row["priority_num"])
        assigned = row["priority_num"]
        if assigned <= 1: return 1 if delta >= 2 else 0
        else:             return 1 if delta >= 1 else 0

    df["mismatch_label"] = df.apply(compute_mismatch, axis=1)

    def mismatch_type(row):
        if row["mismatch_label"] == 0: return "Consistent"
        return "Hidden Crisis" if row["severity_delta"] > 0 else "False Alarm"

    df["mismatch_type"] = df.apply(mismatch_type, axis=1)

    # Signal agreement
    sig1_bin = (df["signal_rule"] > 0.5).astype(int)
    sig2_bin = (df["signal_resolution"] > 0.5).astype(int)
    sig3_bin = (df["signal_satisfaction"] > 0.5).astype(int)

    print("=== Signal Agreement (Cohen's Kappa) ===")
    print(f"Rule vs Resolution:   κ = {cohen_kappa_score(sig1_bin, sig2_bin):.3f}")
    print(f"Rule vs Satisfaction: κ = {cohen_kappa_score(sig1_bin, sig3_bin):.3f}")
    print(f"Mismatch rate: {df['mismatch_label'].mean():.1%}")

    return df

# ── Dataset class ──────────────────────────────────────────────────────────────
def build_model_input(row):
    return (
        f"Subject: {row['Ticket_Subject']} "
        f"[SEP] Description: {row['Ticket_Description']} "
        f"[SEP] Channel: {row['Ticket_Channel']} "
        f"[SEP] Category: {row['Issue_Category']} "
        f"[SEP] Resolution: {row['Resolution_Time_Hours']} hours "
        f"[SEP] Priority: {row['Priority_Level']}"
    )

class TicketDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.texts    = dataframe["model_input"].tolist()
        self.labels   = dataframe["mismatch_label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = outputs.logits.float().argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    recalls  = recall_score(all_labels, all_preds, average=None)
    return acc, macro_f1, recalls

# ── Main training function ─────────────────────────────────────────────────────
def train(data_path, output_dir="final_model", epochs=3, batch_size=16, lr=2e-4):
    print("Loading data...")
    df = pd.read_csv(data_path)

    print("Preprocessing...")
    df = preprocess(df)

    print("Generating pseudo-labels...")
    df = generate_pseudo_labels(df)

    df["model_input"] = df.apply(build_model_input, axis=1)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["mismatch_label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["mismatch_label"]
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    MODEL_NAME = "microsoft/deberta-v3-small"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader = DataLoader(TicketDataset(train_df, tokenizer), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TicketDataset(val_df,   tokenizer), batch_size=32, shuffle=False)
    test_loader  = DataLoader(TicketDataset(test_df,  tokenizer), batch_size=32, shuffle=False)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
        lora_dropout=0.1, bias="none",
        target_modules=["query_proj","value_proj","key_proj","dense"]
    )
    model  = get_peft_model(base_model, lora_config).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.print_trainable_parameters()

    n_consistent = (train_df["mismatch_label"]==0).sum()
    n_mismatch   = (train_df["mismatch_label"]==1).sum()
    total        = len(train_df)
    class_weights = torch.tensor(
        [total/(2*n_consistent), total/(2*n_mismatch)], dtype=torch.float32
    ).to(device)
    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps//10,
        num_training_steps=total_steps
    )

    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        val_acc, val_f1, val_recalls = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
              f"Recall: {val_recalls}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            merged = model.merge_and_unload()
            merged.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  💾 Saved best model (F1: {best_f1:.4f})")

    print("\n=== FINAL TEST RESULTS ===")
    test_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_acc, test_f1, test_recalls = evaluate(test_model, test_loader, device)
    print(f"Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
    print(f"Recall → Consistent: {test_recalls[0]:.4f} | Mismatch: {test_recalls[1]:.4f}")
    print(classification_report(
        [b["label"].item() for loader in [test_loader] for b in loader
         for _ in [None]],
        [], target_names=["Consistent","Mismatch"]
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Training Pipeline")
    parser.add_argument("--data",       type=str, required=True, help="Path to CSV file")
    parser.add_argument("--output_dir", type=str, default="final_model")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=2e-4)
    args = parser.parse_args()
    train(args.data, args.output_dir, args.epochs, args.batch_size, args.lr)
