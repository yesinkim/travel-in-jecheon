"""
Comprehensive RAG Evaluation System

Combines LLM-based metrics (Gemini judge) and automatic metrics (ROUGE, BERTScore)
for complete RAG system evaluation.

Metrics:
1. LLM Judge (Gemini):
   - Accuracy (true/false)
   - Helpfulness (1-10)
   - Relevance (1-10)
   - Depth (1-10)

2. Automatic Metrics:
   - ROUGE-1, ROUGE-2, ROUGE-L
   - BERTScore (P, R, F1)
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics import GenerationMetrics


class ComprehensiveRAGEvaluator:
    """
    Comprehensive evaluator combining LLM judge and automatic metrics
    """

    def __init__(self, gemini_model: str = "gemini-2.0-flash-exp"):
        """
        Initialize evaluator with both LLM judge and automatic metrics

        Args:
            gemini_model: Gemini model to use for LLM judge
        """
        self.llm_judge = LLMJudge(model_name=gemini_model)
        self.generation_metrics = GenerationMetrics()

    def load_ground_truth(self, qa_pairs_path: str) -> Dict[str, str]:
        """
        Load ground truth answers from qa_pairs.jsonl

        Args:
            qa_pairs_path: Path to qa_pairs.jsonl file

        Returns:
            Dictionary mapping questions to ground truth answers
        """
        ground_truth = {}

        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa = json.loads(line.strip())
                ground_truth[qa['question']] = qa['answer']

        return ground_truth

    def evaluate_single_response(
        self,
        question: str,
        response: str,
        ground_truth: str,
        retrieved_docs: List[Dict],
        documents: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate a single response with all metrics

        Args:
            question: User question
            response: Model's generated answer
            ground_truth: Ground truth answer
            retrieved_docs: Retrieved document metadata
            documents: All documents dictionary

        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}

        # 1. LLM Judge Metrics
        print("    Evaluating with LLM judge...")
        llm_metrics = self.llm_judge.evaluate_response(
            question=question,
            response=response,
            retrieved_docs=retrieved_docs,
            documents=documents
        )
        results.update(llm_metrics)

        # 2. Automatic Generation Metrics
        print("    Calculating ROUGE scores...")
        rouge_scores = self.generation_metrics.rouge_scores(
            prediction=response,
            reference=ground_truth
        )
        results.update(rouge_scores)

        # 3. BERTScore (single pair)
        print("    Calculating BERTScore...")
        bert_scores = self.generation_metrics.bert_score(
            predictions=[response],
            references=[ground_truth],
            lang="ko"
        )
        results.update(bert_scores)

        return results

    def evaluate_rag_comparison(
        self,
        rag_results_path: str,
        documents_path: str,
        qa_pairs_path: str,
        output_csv_path: str
    ) -> pd.DataFrame:
        """
        Evaluate RAG comparison results with comprehensive metrics

        Args:
            rag_results_path: Path to rag_comparison.json
            documents_path: Path to documents.jsonl
            qa_pairs_path: Path to qa_pairs.jsonl
            output_csv_path: Path to save CSV output

        Returns:
            DataFrame with all evaluation results
        """
        # Load data
        print("Loading data...")
        documents = self.llm_judge.load_documents(documents_path)
        ground_truth = self.load_ground_truth(qa_pairs_path)

        with open(rag_results_path, 'r', encoding='utf-8') as f:
            rag_results = json.load(f)

        print(f"Loaded {len(rag_results)} test cases")
        print(f"Loaded {len(ground_truth)} ground truth answers")

        # Evaluate each test case
        all_results = []

        for i, result in enumerate(tqdm(rag_results, desc="Evaluating")):
            question = result["question"]

            # Get ground truth
            gt_answer = ground_truth.get(question)
            if not gt_answer:
                print(f"\n⚠️  Warning: No ground truth for question: {question}")
                continue

            print(f"\n{'='*80}")
            print(f"Question {i+1}/{len(rag_results)}: {question}")
            print(f"{'='*80}")

            # Evaluate each model
            for model_name, model_result in result["models"].items():
                print(f"\n🤖 Evaluating {model_name}...")

                try:
                    # Evaluate with all metrics
                    metrics = self.evaluate_single_response(
                        question=question,
                        response=model_result["answer"],
                        ground_truth=gt_answer,
                        retrieved_docs=model_result["retrieved_docs"],
                        documents=documents
                    )

                    # Compile results
                    row = {
                        "question": question,
                        "model": model_name,
                        "ground_truth": gt_answer,
                        "generated_answer": model_result["answer"],
                        "retrieved_docs": ", ".join([d["title"] for d in model_result["retrieved_docs"]]),

                        # LLM Judge Metrics
                        "accuracy": metrics.get("accuracy"),
                        "accuracy_explanation": metrics.get("accuracy_explanation", ""),
                        "helpfulness": metrics.get("helpfulness"),
                        "helpfulness_explanation": metrics.get("helpfulness_explanation", ""),
                        "relevance": metrics.get("relevance"),
                        "relevance_explanation": metrics.get("relevance_explanation", ""),
                        "depth": metrics.get("depth"),
                        "depth_explanation": metrics.get("depth_explanation", ""),

                        # Automatic Metrics
                        "rouge1": metrics.get("rouge1"),
                        "rouge2": metrics.get("rouge2"),
                        "rougeL": metrics.get("rougeL"),
                        "bert_precision": metrics.get("bert_precision"),
                        "bert_recall": metrics.get("bert_recall"),
                        "bert_f1": metrics.get("bert_f1"),
                    }

                    all_results.append(row)

                    # Print summary
                    print(f"  ✓ Accuracy: {metrics.get('accuracy', 'N/A')}")
                    print(f"  ✓ Helpfulness: {metrics.get('helpfulness', 'N/A')}/10")
                    print(f"  ✓ Relevance: {metrics.get('relevance', 'N/A')}/10")
                    print(f"  ✓ Depth: {metrics.get('depth', 'N/A')}/10")
                    print(f"  ✓ ROUGE-1: {metrics.get('rouge1', 0):.4f}")
                    print(f"  ✓ ROUGE-L: {metrics.get('rougeL', 0):.4f}")
                    print(f"  ✓ BERTScore F1: {metrics.get('bert_f1', 0):.4f}")

                except Exception as e:
                    print(f"  ❌ Error evaluating {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Create DataFrame
        df = pd.DataFrame(all_results)

        # Save to CSV
        print(f"\n{'='*80}")
        print(f"Saving results to {output_csv_path}...")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ Results saved!")

        # Print summary statistics
        self.print_summary_statistics(df)

        return df

    def print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics by model"""
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS BY MODEL")
        print(f"{'='*80}\n")

        for model in df['model'].unique():
            model_df = df[df['model'] == model]

            print(f"🤖 {model}")
            print(f"{'─'*80}")

            # LLM Judge Metrics
            print("\n📊 LLM Judge Metrics:")
            accuracy_rate = model_df['accuracy'].mean() if 'accuracy' in model_df.columns else 0
            print(f"  Accuracy Rate:     {accuracy_rate*100:.1f}%")
            print(f"  Avg Helpfulness:   {model_df['helpfulness'].mean():.2f}/10")
            print(f"  Avg Relevance:     {model_df['relevance'].mean():.2f}/10")
            print(f"  Avg Depth:         {model_df['depth'].mean():.2f}/10")

            # Automatic Metrics
            print("\n📈 Automatic Metrics:")
            print(f"  ROUGE-1:           {model_df['rouge1'].mean():.4f}")
            print(f"  ROUGE-2:           {model_df['rouge2'].mean():.4f}")
            print(f"  ROUGE-L:           {model_df['rougeL'].mean():.4f}")
            print(f"  BERTScore P:       {model_df['bert_precision'].mean():.4f}")
            print(f"  BERTScore R:       {model_df['bert_recall'].mean():.4f}")
            print(f"  BERTScore F1:      {model_df['bert_f1'].mean():.4f}")

            print(f"\n  Total Samples: {len(model_df)}")
            print(f"{'─'*80}\n")

        # Model comparison
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}\n")

        comparison = df.groupby('model').agg({
            'accuracy': 'mean',
            'helpfulness': 'mean',
            'relevance': 'mean',
            'depth': 'mean',
            'rouge1': 'mean',
            'rougeL': 'mean',
            'bert_f1': 'mean'
        }).round(4)

        print(comparison.to_string())
        print()


def evaluate_rag_with_all_metrics(
    rag_results_path: str = "notebooks/results/rag_comparison.json",
    documents_path: str = "data/chunks/documents.jsonl",
    qa_pairs_path: str = "data/chunks/qa_pairs.jsonl",
    output_csv_path: str = "results/evaluation/comprehensive_evaluation.csv"
) -> pd.DataFrame:
    """
    Convenience function to run comprehensive evaluation

    Args:
        rag_results_path: Path to RAG comparison results
        documents_path: Path to documents
        qa_pairs_path: Path to QA pairs with ground truth
        output_csv_path: Path to save CSV output

    Returns:
        DataFrame with all evaluation results
    """
    evaluator = ComprehensiveRAGEvaluator()

    return evaluator.evaluate_rag_comparison(
        rag_results_path=rag_results_path,
        documents_path=documents_path,
        qa_pairs_path=qa_pairs_path,
        output_csv_path=output_csv_path
    )


if __name__ == "__main__":
    # Run comprehensive evaluation
    df = evaluate_rag_with_all_metrics()

    print("\n✅ Comprehensive evaluation complete!")
    print(f"📊 Results shape: {df.shape}")
    print(f"📊 Models evaluated: {df['model'].unique()}")
