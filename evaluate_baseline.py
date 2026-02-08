"""
Evaluate Baseline Predictions Against HuggingFace QEvasion Ground Truth

This script takes baseline_predictions.csv and compares the predicted_label
against the ground truth labels from the ailsntua/QEvasion HuggingFace dataset.

USAGE:
    python evaluate_baseline.py
    python evaluate_baseline.py --predictions baseline_predictions.csv --output results.csv
"""

import argparse
import re
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# HuggingFace dataset config
HF_DATASET = "ailsntua/QEvasion"
HF_TEST_SPLIT = "data/test-00000-of-00001.parquet"


def load_ground_truth() -> pd.DataFrame:
    """Load the QEvasion test dataset from HuggingFace."""
    print(f"Loading ground truth from HuggingFace: {HF_DATASET}")
    df = pd.read_parquet(f"hf://datasets/{HF_DATASET}/{HF_TEST_SPLIT}")
    print(f"  Loaded {len(df)} examples")
    return df


def load_predictions(path: str) -> pd.DataFrame:
    """Load predictions from CSV file."""
    print(f"Loading predictions from: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} predictions")
    return df


def extract_label_from_prediction(text: str) -> str:
    """
    Extract label from raw prediction text.
    The format is: "analysis...assistantfinalLabel" (no space before label)
    """
    if pd.isna(text) or text == "":
        return "MISSING"
    
    text = str(text).strip()
    
    # Check if it's already a clean label
    clean_labels = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]
    if text in clean_labels:
        return text
    
    label_map = {
        "clear reply": "Clear Reply",
        "clear non-reply": "Clear Non-Reply", 
        "ambivalent": "Ambivalent"
    }
    
    # Try to find "assistantfinal" pattern (specific to this baseline format)
    # Pattern: assistantfinal followed directly by the label (no space)
    match = re.search(r'assistantfinal(Clear Reply|Clear Non-Reply|Ambivalent)\s*$', text, re.IGNORECASE)
    if match:
        return label_map.get(match.group(1).lower(), match.group(1))
    
    # Also try with possible whitespace
    match = re.search(r'assistantfinal\s*(Clear Reply|Clear Non-Reply|Ambivalent)', text, re.IGNORECASE)
    if match:
        return label_map.get(match.group(1).lower(), match.group(1))
    
    # Try to find LABEL: pattern
    match = re.search(r'LABEL:\s*(Clear Reply|Clear Non-Reply|Ambivalent)', text, re.IGNORECASE)
    if match:
        return label_map.get(match.group(1).lower(), match.group(1))
    
    # Fallback: check last 50 chars for the final label (more precise)
    last_part = text[-50:].lower()
    if last_part.endswith("ambivalent"):
        return "Ambivalent"
    elif last_part.endswith("clear non-reply"):
        return "Clear Non-Reply"
    elif last_part.endswith("clear reply"):
        return "Clear Reply"
    
    return "PARSE_ERROR"


def normalize_label(label: str) -> str:
    """Normalize label for comparison."""
    if pd.isna(label) or label == "":
        return "MISSING"
    
    label = str(label).strip().lower()
    
    if "clear reply" in label and "non" not in label:
        return "Clear Reply"
    elif "non-reply" in label or "non reply" in label:
        return "Clear Non-Reply"
    elif "ambivalent" in label:
        return "Ambivalent"
    
    return "UNKNOWN"


def evaluate(y_true: list, y_pred: list) -> dict:
    """Compute evaluation metrics."""
    labels = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]
    
    # Filter to valid predictions only
    valid_mask = [
        (t in labels) and (p in labels)
        for t, p in zip(y_true, y_pred)
    ]
    
    y_true_valid = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]
    
    if len(y_true_valid) == 0:
        return {"accuracy": 0.0, "valid_count": 0, "total_count": len(y_true)}
    
    return {
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "precision_macro": precision_score(y_true_valid, y_pred_valid, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_valid, y_pred_valid, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_valid, y_pred_valid, labels=labels, average="macro", zero_division=0),
        "valid_count": len(y_true_valid),
        "total_count": len(y_true),
        "y_true": y_true_valid,
        "y_pred": y_pred_valid,
    }


def print_results(metrics: dict):
    """Print evaluation results."""
    labels = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nTotal examples: {metrics['total_count']}")
    print(f"Valid predictions: {metrics['valid_count']}")
    print(f"Invalid/Parse errors: {metrics['total_count'] - metrics['valid_count']}")
    
    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 Score (macro):  {metrics['f1_macro']:.4f}")
    
    # Classification report
    if metrics["valid_count"] > 0:
        print("\n" + "-" * 60)
        print("CLASSIFICATION REPORT")
        print("-" * 60)
        print(classification_report(
            metrics["y_true"],
            metrics["y_pred"],
            labels=labels,
            zero_division=0
        ))
        
        # Confusion matrix
        print("\n" + "-" * 60)
        print("CONFUSION MATRIX")
        print("-" * 60)
        print("                      Predicted")
        print("                      " + "  ".join([f"{l[:10]:>12}" for l in labels]))
        cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=labels)
        for i, label in enumerate(labels):
            row = "  ".join([f"{v:>12}" for v in cm[i]])
            print(f"Actual {label:<14} {row}")
    
    # Prediction distribution
    print("\n" + "-" * 60)
    print("PREDICTION DISTRIBUTION")
    print("-" * 60)
    
    for label in labels:
        gt_count = sum(1 for t in metrics["y_true"] if t == label)
        pred_count = sum(1 for p in metrics["y_pred"] if p == label)
        print(f"  {label}:")
        print(f"    Ground Truth: {gt_count} ({gt_count/len(metrics['y_true'])*100:.1f}%)")
        print(f"    Predicted:    {pred_count} ({pred_count/len(metrics['y_pred'])*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline predictions against HuggingFace QEvasion ground truth"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="baseline_predictions.csv",
        help="Path to predictions CSV file (default: baseline_predictions.csv)"
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="prediction",
        help="Column name containing raw predictions to parse (default: 'prediction')"
    )
    parser.add_argument(
        "--use-extracted",
        action="store_true",
        help="Use pre-extracted 'predicted_label' column instead of parsing from raw"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for detailed results CSV (optional)"
    )
    args = parser.parse_args()
    
    # Load data
    pred_df = load_predictions(args.predictions)
    gt_df = load_ground_truth()
    
    # Align datasets by minimum length
    min_len = min(len(pred_df), len(gt_df))
    print(f"\nUsing first {min_len} examples for evaluation")
    
    pred_df = pred_df.head(min_len).reset_index(drop=True)
    gt_df = gt_df.head(min_len).reset_index(drop=True)
    
    # Extract predictions
    if args.use_extracted:
        # Use pre-extracted predicted_label column (may have errors)
        if "predicted_label" not in pred_df.columns:
            print(f"\nError: 'predicted_label' column not found")
            print(f"Available columns: {list(pred_df.columns)}")
            return
        print(f"\nUsing pre-extracted 'predicted_label' column...")
        y_pred = [normalize_label(p) for p in pred_df["predicted_label"]]
    else:
        # Parse from raw prediction text (default - more accurate)
        col = args.prediction_column
        if col not in pred_df.columns:
            print(f"\nError: Column '{col}' not found in predictions CSV")
            print(f"Available columns: {list(pred_df.columns)}")
            return
        print(f"\nParsing labels from '{col}' column...")
        y_pred = [extract_label_from_prediction(p) for p in pred_df[col]]
    
    # Extract ground truth from HuggingFace dataset
    print("Extracting ground truth labels from HuggingFace dataset...")
    y_true = [normalize_label(gt) for gt in gt_df["clarity_label"]]
    
    # Count parse errors
    parse_errors = sum(1 for p in y_pred if p in ["PARSE_ERROR", "MISSING", "UNKNOWN"])
    if parse_errors > 0:
        print(f"\nWarning: {parse_errors} predictions could not be parsed")
    
    # Evaluate
    metrics = evaluate(y_true, y_pred)
    
    # Print results
    print_results(metrics)
    
    # Save detailed results if requested
    if args.output:
        results_df = pd.DataFrame({
            "index": range(min_len),
            "question": gt_df["question"] if "question" in gt_df.columns else "",
            "ground_truth": y_true,
            "prediction": y_pred,
            "correct": [t == p for t, p in zip(y_true, y_pred)],
        })
        results_df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
