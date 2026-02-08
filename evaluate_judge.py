"""
Evaluate Judge Predictions Against Ground Truth Labels

This script compares the judge's final predictions with the actual clarity labels
and computes accuracy metrics for all models involved.
"""

import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(
    judge_output_path: str,
    ground_truth_path: str,
) -> pd.DataFrame:
    """Load and merge judge output with ground truth labels."""
    print("\n" + "=" * 60)
    print("Loading data files...")
    print("=" * 60)
    
    # Load judge output
    judge_df = pd.read_csv(judge_output_path)
    print(f"  Judge output: {len(judge_df)} rows from {judge_output_path}")
    
    # Load ground truth (handle both xlsx and csv)
    if ground_truth_path.endswith('.xlsx'):
        gt_df = pd.read_excel(ground_truth_path)
    else:
        gt_df = pd.read_csv(ground_truth_path)
    print(f"  Ground truth: {len(gt_df)} rows from {ground_truth_path}")
    
    # Align by row count
    min_len = min(len(judge_df), len(gt_df))
    print(f"\n  Using first {min_len} rows")
    
    judge_df = judge_df.head(min_len).reset_index(drop=True)
    gt_df = gt_df.head(min_len).reset_index(drop=True)
    
    # Add ground truth label to judge dataframe
    if 'clarity_label' in gt_df.columns:
        judge_df['ground_truth'] = gt_df['clarity_label']
    else:
        raise ValueError("Ground truth file must have 'clarity_label' column")
    
    return judge_df


def normalize_label(label: str) -> str:
    """Normalize label for comparison."""
    if pd.isna(label) or label == '':
        return 'MISSING'
    label = str(label).strip().lower()
    if 'clear reply' in label and 'non' not in label:
        return 'Clear Reply'
    elif 'non-reply' in label or 'non reply' in label:
        return 'Clear Non-Reply'
    elif 'ambivalent' in label:
        return 'Ambivalent'
    return 'UNKNOWN'


def compute_metrics(y_true: list, y_pred: list, name: str) -> dict:
    """Compute accuracy metrics for a set of predictions."""
    # Filter out missing/unknown labels
    valid_mask = [(t not in ['MISSING', 'UNKNOWN'] and p not in ['MISSING', 'UNKNOWN', 'N/A', 'ERROR', 'PARSE_ERROR']) 
                  for t, p in zip(y_true, y_pred)]
    
    y_true_valid = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]
    
    if len(y_true_valid) == 0:
        return {'accuracy': 0.0, 'valid_count': 0, 'total_count': len(y_true)}
    
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    
    return {
        'accuracy': accuracy,
        'valid_count': len(y_true_valid),
        'total_count': len(y_true),
        'y_true': y_true_valid,
        'y_pred': y_pred_valid,
    }


def print_evaluation(df: pd.DataFrame):
    """Print comprehensive evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Normalize all labels
    df['ground_truth_norm'] = df['ground_truth'].apply(normalize_label)
    df['model1_norm'] = df['model1_prediction'].apply(normalize_label)
    df['model2_norm'] = df['model2_prediction'].apply(normalize_label)
    df['final_norm'] = df['final_prediction'].apply(normalize_label)
    
    # Filter to valid ground truth only
    valid_gt_mask = df['ground_truth_norm'].isin(['Clear Reply', 'Clear Non-Reply', 'Ambivalent'])
    df_valid = df[valid_gt_mask].copy()
    
    print(f"\nTotal examples: {len(df)}")
    print(f"Examples with valid ground truth: {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("\nNo valid ground truth labels found for evaluation!")
        return
    
    # Compute metrics for each model
    labels = ['Clear Reply', 'Clear Non-Reply', 'Ambivalent']
    
    print("\n" + "-" * 60)
    print("ACCURACY COMPARISON")
    print("-" * 60)
    
    # Model 1
    m1_metrics = compute_metrics(
        df_valid['ground_truth_norm'].tolist(),
        df_valid['model1_norm'].tolist(),
        "Model 1"
    )
    print(f"\nModel 1 (clarity_predictions_with_labels):")
    print(f"  Accuracy: {m1_metrics['accuracy']:.4f} ({m1_metrics['accuracy']*100:.2f}%)")
    print(f"  Valid predictions: {m1_metrics['valid_count']}/{m1_metrics['total_count']}")
    
    # Model 2
    m2_metrics = compute_metrics(
        df_valid['ground_truth_norm'].tolist(),
        df_valid['model2_norm'].tolist(),
        "Model 2"
    )
    print(f"\nModel 2 (qwen3_mtl_test_predictions):")
    print(f"  Accuracy: {m2_metrics['accuracy']:.4f} ({m2_metrics['accuracy']*100:.2f}%)")
    print(f"  Valid predictions: {m2_metrics['valid_count']}/{m2_metrics['total_count']}")
    
    # Final (Judge-resolved)
    final_metrics = compute_metrics(
        df_valid['ground_truth_norm'].tolist(),
        df_valid['final_norm'].tolist(),
        "Final (Judge)"
    )
    print(f"\nFinal (Judge-resolved):")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"  Valid predictions: {final_metrics['valid_count']}/{final_metrics['total_count']}")
    
    # Improvement analysis
    print("\n" + "-" * 60)
    print("IMPROVEMENT ANALYSIS")
    print("-" * 60)
    
    best_baseline = max(m1_metrics['accuracy'], m2_metrics['accuracy'])
    improvement = final_metrics['accuracy'] - best_baseline
    print(f"\nBest baseline accuracy: {best_baseline:.4f}")
    print(f"Judge-resolved accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Disagreement analysis
    disagreements = df_valid[df_valid['model1_norm'] != df_valid['model2_norm']]
    print(f"\nDisagreements: {len(disagreements)}/{len(df_valid)} ({len(disagreements)/len(df_valid)*100:.1f}%)")
    
    if len(disagreements) > 0:
        # How often was each model correct on disagreements?
        m1_correct_on_disagree = sum(disagreements['model1_norm'] == disagreements['ground_truth_norm'])
        m2_correct_on_disagree = sum(disagreements['model2_norm'] == disagreements['ground_truth_norm'])
        judge_correct_on_disagree = sum(disagreements['final_norm'] == disagreements['ground_truth_norm'])
        
        print(f"\nOn disagreements ({len(disagreements)} cases):")
        print(f"  Model 1 correct: {m1_correct_on_disagree} ({m1_correct_on_disagree/len(disagreements)*100:.1f}%)")
        print(f"  Model 2 correct: {m2_correct_on_disagree} ({m2_correct_on_disagree/len(disagreements)*100:.1f}%)")
        print(f"  Judge correct: {judge_correct_on_disagree} ({judge_correct_on_disagree/len(disagreements)*100:.1f}%)")
    
    # Detailed classification report for final predictions
    if final_metrics['valid_count'] > 0:
        print("\n" + "-" * 60)
        print("CLASSIFICATION REPORT (Final Judge-resolved)")
        print("-" * 60)
        print(classification_report(
            final_metrics['y_true'],
            final_metrics['y_pred'],
            labels=labels,
            zero_division=0
        ))
        
        # Confusion matrix
        print("\n" + "-" * 60)
        print("CONFUSION MATRIX (Final Judge-resolved)")
        print("-" * 60)
        print("                      Predicted")
        print("                      " + "  ".join([f"{l[:8]:>10}" for l in labels]))
        cm = confusion_matrix(final_metrics['y_true'], final_metrics['y_pred'], labels=labels)
        for i, label in enumerate(labels):
            row = "  ".join([f"{v:>10}" for v in cm[i]])
            print(f"Actual {label[:12]:<12} {row}")
    
    return df_valid


def save_detailed_results(df: pd.DataFrame, output_path: str):
    """Save detailed evaluation results with correctness flags."""
    df = df.copy()
    
    # Add correctness columns
    df['model1_correct'] = df['model1_norm'] == df['ground_truth_norm']
    df['model2_correct'] = df['model2_norm'] == df['ground_truth_norm']
    df['final_correct'] = df['final_norm'] == df['ground_truth_norm']
    df['models_disagree'] = df['model1_norm'] != df['model2_norm']
    
    # Select output columns
    output_cols = [
        'index', 'question',
        'ground_truth', 'model1_prediction', 'model2_prediction', 'final_prediction',
        'model1_correct', 'model2_correct', 'final_correct', 'models_disagree',
        'judge_prediction', 'judge_reasoning'
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    
    df[output_cols].to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate judge predictions against ground truth labels"
    )
    parser.add_argument(
        "--judge-output",
        type=str,
        default="judge_resolved_predictions.csv",
        help="Path to judge output CSV"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="llama.csv",
        help="Path to ground truth file (xlsx or csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_detailed.csv",
        help="Output path for detailed results"
    )
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.judge_output, args.ground_truth)
    
    # Print evaluation
    df_eval = print_evaluation(df)
    
    # Save detailed results
    if df_eval is not None and len(df_eval) > 0:
        save_detailed_results(df_eval, args.output)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
