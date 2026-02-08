"""
Cosine Similarity Analysis for QEvasion Dataset

Computes cosine similarity between interview questions and answers
to identify low-similarity pairs using sentence embeddings.
"""

import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# HuggingFace dataset config
HF_DATASET = "ailsntua/QEvasion"
HF_TEST_SPLIT = "data/test-00000-of-00001.parquet"


def load_test_dataset() -> pd.DataFrame:
    """Load the QEvasion test dataset from HuggingFace."""
    print(f"Loading dataset from HuggingFace: {HF_DATASET}")
    df = pd.read_parquet(f"hf://datasets/{HF_DATASET}/{HF_TEST_SPLIT}")
    print(f"Loaded {len(df)} examples from test dataset")
    return df


def compute_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
    desc: str = "Encoding",
) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    # Clean texts - replace NaN with empty string
    cleaned_texts = [str(t) if pd.notna(t) else "" for t in texts]
    
    embeddings = model.encode(
        cleaned_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  {desc}: {len(embeddings)} texts encoded")
    return embeddings


def calculate_similarity(
    question_embeddings: np.ndarray,
    answer_embeddings: np.ndarray,
) -> np.ndarray:
    """Calculate pairwise cosine similarity between questions and answers."""
    # Calculate row-wise cosine similarity (each question with its corresponding answer)
    similarities = []
    for q_emb, a_emb in zip(question_embeddings, answer_embeddings):
        sim = cosine_similarity([q_emb], [a_emb])[0][0]
        similarities.append(sim)
    return np.array(similarities)


def analyze_results(
    df: pd.DataFrame,
    similarities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Analyze similarity results and print statistics."""
    # Add similarity scores to dataframe
    df = df.copy()
    df['cosine_similarity'] = similarities
    
    print("\n" + "=" * 60)
    print("SIMILARITY STATISTICS")
    print("=" * 60)
    print(f"\n  Mean:   {np.mean(similarities):.4f}")
    print(f"  Median: {np.median(similarities):.4f}")
    print(f"  Std:    {np.std(similarities):.4f}")
    print(f"  Min:    {np.min(similarities):.4f}")
    print(f"  Max:    {np.max(similarities):.4f}")
    
    # Count low similarity pairs
    low_sim_mask = similarities < threshold
    low_sim_count = np.sum(low_sim_mask)
    low_sim_pct = low_sim_count / len(similarities) * 100
    
    print(f"\n  Low Similarity (< {threshold}): {low_sim_count} examples ({low_sim_pct:.1f}%)")
    
    # Distribution by bins
    print("\n  Similarity Distribution:")
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in bins:
        count = np.sum((similarities >= low) & (similarities < high))
        pct = count / len(similarities) * 100
        bar = "#" * int(pct / 2)
        print(f"    [{low:.1f}-{high:.1f}): {count:4d} ({pct:5.1f}%) {bar}")
    
    # Show examples of low similarity pairs
    if low_sim_count > 0:
        print("\n" + "=" * 60)
        print(f"EXAMPLES OF LOW SIMILARITY PAIRS (< {threshold})")
        print("=" * 60)
        
        low_sim_df = df[low_sim_mask].sort_values('cosine_similarity')
        
        for idx, row in low_sim_df.head(5).iterrows():
            print(f"\n--- Example (similarity: {row['cosine_similarity']:.4f}) ---")
            print(f"Question: {str(row['interview_question'])[:200]}...")
            print(f"Answer: {str(row['interview_answer'])[:200]}...")
            if 'clarity_label' in row:
                print(f"Label: {row.get('clarity_label', 'N/A')}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cosine similarity between questions and answers in QEvasion dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model to use (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Similarity threshold for 'low similarity' (default: 0.3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="similarity_analysis.csv",
        help="Output CSV path (default: similarity_analysis.csv)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to local CSV file (if not provided, loads from HuggingFace)"
    )
    args = parser.parse_args()
    
    # Load dataset
    if args.data:
        print(f"Loading dataset from: {args.data}")
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df)} examples")
    else:
        df = load_test_dataset()
    
    # Load sentence transformer model
    print(f"\nLoading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"Model loaded (embedding dimension: {model.get_sentence_embedding_dimension()})")
    
    # Extract texts
    questions = df['question'].tolist()
    answers = df['interview_answer'].tolist()
    
    # Compute embeddings
    print("\nComputing embeddings...")
    question_embeddings = compute_embeddings(
        model, questions, batch_size=args.batch_size, desc="Questions encoded"
    )
    answer_embeddings = compute_embeddings(
        model, answers, batch_size=args.batch_size, desc="Answers encoded"
    )
    
    # Calculate similarities
    print("\nCalculating cosine similarities...")
    similarities = calculate_similarity(question_embeddings, answer_embeddings)
    
    # Analyze and get results
    results_df = analyze_results(df, similarities, args.threshold)
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Return summary for programmatic use
    low_sim_count = np.sum(similarities < args.threshold)
    return {
        "total": len(similarities),
        "low_similarity_count": low_sim_count,
        "low_similarity_pct": low_sim_count / len(similarities) * 100,
        "mean": np.mean(similarities),
        "median": np.median(similarities),
    }


if __name__ == "__main__":
    main()
