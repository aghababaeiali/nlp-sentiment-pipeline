"""
evaluate.py
Evaluates the fine-tuned DistilBERT sentiment model on the IMDB test set.
Generates: confusion matrix, classification report, error analysis.

Run from project root:
    python evaluate.py

Requirements:
    pip install scikit-learn matplotlib seaborn datasets transformers torch
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ──────────────────────────────────────────────────────────────
MODEL_ID     = "aliabbi/sentiment-distilbert"
NUM_SAMPLES  = 1000           # test on 1000 samples (fast, representative)
BATCH_SIZE   = 32
MAX_LENGTH   = 256
CM_PATH      = "evaluation_outputs/confusion_matrix.png"
ERRORS_PATH  = "evaluation_outputs/misclassified_examples.csv"

LABEL_MAP = {0: "negative", 1: "positive"}


def load_model():
    print(f"🔄 Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def predict_batch(texts, tokenizer, model):
    """Run batch prediction. Returns (predictions, confidences)."""
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        confs = torch.max(probs, dim=1).values.cpu().numpy()
    return preds, confs


def evaluate(tokenizer, model, samples):
    """Run evaluation across all samples, return predictions + metadata."""
    all_preds = []
    all_confs = []
    all_labels = []
    all_texts = []

    print(f"🔄 Evaluating on {len(samples)} samples...")

    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i+BATCH_SIZE]
        texts = [s["text"] for s in batch]
        labels = [s["label"] for s in batch]

        preds, confs = predict_batch(texts, tokenizer, model)

        all_preds.extend(preds.tolist())
        all_confs.extend(confs.tolist())
        all_labels.extend(labels)
        all_texts.extend(texts)

        if (i // BATCH_SIZE + 1) % 5 == 0:
            print(f"   Progress: {i + len(batch)}/{len(samples)}")

    return all_preds, all_confs, all_labels, all_texts


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Save confusion matrix as PNG."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar=False,
        annot_kws={"size": 16}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"Confusion Matrix — DistilBERT on IMDB ({len(y_true)} samples)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved to: {output_path}")


def error_analysis(preds, confs, labels, texts, output_path):
    """Save misclassified examples to CSV for inspection."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    errors = []
    for pred, conf, label, text in zip(preds, confs, labels, texts):
        if pred != label:
            errors.append({
                "true_label": LABEL_MAP[label],
                "predicted_label": LABEL_MAP[pred],
                "confidence": round(conf, 3),
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            })

    df = pd.DataFrame(errors)
    df = df.sort_values("confidence", ascending=False)  # most confident wrong predictions first
    df.to_csv(output_path, index=False)

    return df


def analyze_error_patterns(errors_df, preds, labels):
    """Print patterns about the errors."""
    print("\n" + "="*60)
    print("  ERROR ANALYSIS")
    print("="*60)

    total_errors = len(errors_df)
    print(f"\n  Total misclassifications: {total_errors}")

    if total_errors == 0:
        print("  Perfect classification! No errors.")
        return

    # False positives vs false negatives
    false_positives = len(errors_df[errors_df["predicted_label"] == "positive"])
    false_negatives = len(errors_df[errors_df["predicted_label"] == "negative"])

    print(f"\n  False Positives (negative → predicted positive): {false_positives}")
    print(f"  False Negatives (positive → predicted negative): {false_negatives}")

    # Confidence distribution
    high_conf_errors = len(errors_df[errors_df["confidence"] > 0.9])
    mid_conf_errors  = len(errors_df[(errors_df["confidence"] > 0.7) & (errors_df["confidence"] <= 0.9)])
    low_conf_errors  = len(errors_df[errors_df["confidence"] <= 0.7])

    print(f"\n  Confidence breakdown of errors:")
    print(f"    High confidence (>0.9): {high_conf_errors}  ← model was wrong AND confident")
    print(f"    Mid confidence (0.7-0.9): {mid_conf_errors}")
    print(f"    Low confidence (<0.7): {low_conf_errors}    ← uncertainty detected")

    # Show top 3 most confident wrong predictions
    print("\n  Top 3 most confident misclassifications:")
    for i, row in errors_df.head(3).iterrows():
        print(f"\n    Confidence: {row['confidence']:.2%}")
        print(f"    True: {row['true_label']}  →  Predicted: {row['predicted_label']}")
        print(f"    Text: {row['text_preview'][:150]}...")


def main():
    print("\n📊 Sentiment Analysis — Evaluation on IMDB Test Set")
    print(f"   Samples: {NUM_SAMPLES}\n")

    # Load test data
    print("🔄 Loading IMDB test set...")
    dataset = load_dataset("imdb", split="test")
    dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
    samples = list(dataset)

    # Load model
    tokenizer, model = load_model()

    # Run evaluation
    preds, confs, labels, texts = evaluate(tokenizer, model, samples)

    # Metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision, recall, _, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    print("\n" + "="*60)
    print("  OVERALL METRICS")
    print("="*60)
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")

    # Classification report
    print("\n" + "="*60)
    print("  PER-CLASS METRICS")
    print("="*60)
    print(classification_report(labels, preds, target_names=["negative", "positive"], digits=4))

    # Confusion matrix
    plot_confusion_matrix(labels, preds, CM_PATH)

    # Error analysis
    errors_df = error_analysis(preds, confs, labels, texts, ERRORS_PATH)
    analyze_error_patterns(errors_df, preds, labels)

    print(f"\n✅ Full misclassified examples saved to: {ERRORS_PATH}")
    print(f"   Inspect them to understand what the model gets wrong.\n")


if __name__ == "__main__":
    main()