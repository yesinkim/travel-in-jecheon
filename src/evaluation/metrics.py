"""
RAG Evaluation Metrics Module

This module provides evaluation metrics for RAG systems, including:
- Retrieval metrics (Recall@K, MRR, NDCG)
- Generation metrics (ROUGE, BERTScore)
- RAG-specific metrics (Faithfulness, Context/Answer Relevance)
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict


class RetrievalMetrics:
    """Metrics for evaluating document retrieval quality"""

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
        """
        Calculate Recall@K metric

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by relevance)
            relevant_docs: List of ground truth relevant document IDs
            k: Number of top documents to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 0.0

        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)

        hits = len(retrieved_k & relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by relevance)
            relevant_docs: List of ground truth relevant document IDs

        Returns:
            MRR score
        """
        relevant_set = set(relevant_docs)

        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 3) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by relevance)
            relevant_docs: List of ground truth relevant document IDs
            k: Number of top documents to consider

        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 0.0

        relevant_set = set(relevant_docs)

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k], start=1):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 1)

        # Calculate IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), k)))

        return dcg / idcg if idcg > 0 else 0.0


class GenerationMetrics:
    """Metrics for evaluating text generation quality"""

    def __init__(self):
        self._rouge_scorer = None
        self._bert_scorer = None

    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)

        Args:
            prediction: Generated answer
            reference: Ground truth answer

        Returns:
            Dictionary with ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer

            if self._rouge_scorer is None:
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'],
                    use_stemmer=True
                )

            scores = self._rouge_scorer.score(reference, prediction)

            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure,
            }
        except ImportError:
            print("Warning: rouge_score not installed. Install with: pip install rouge-score")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def bert_score(self, predictions: List[str], references: List[str],
                   lang: str = "ko") -> Dict[str, float]:
        """
        Calculate BERTScore

        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            lang: Language code (default: "ko" for Korean)

        Returns:
            Dictionary with average BERTScore (P, R, F1)
        """
        try:
            from bert_score import score

            P, R, F1 = score(
                predictions,
                references,
                lang=lang,
                verbose=False,
                rescale_with_baseline=True
            )

            return {
                'bert_precision': P.mean().item(),
                'bert_recall': R.mean().item(),
                'bert_f1': F1.mean().item(),
            }
        except ImportError:
            print("Warning: bert_score not installed. Install with: pip install bert-score")
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}

    @staticmethod
    def exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
        """
        Calculate Exact Match score

        Args:
            prediction: Generated answer
            reference: Ground truth answer
            normalize: Whether to normalize text (lowercase, strip whitespace)

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if normalize:
            prediction = prediction.lower().strip()
            reference = reference.lower().strip()

        return 1.0 if prediction == reference else 0.0


class RAGMetrics:
    """RAG-specific evaluation metrics"""

    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()

    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        retrieved_docs_list: List[List[str]],
        relevant_docs_list: List[List[str]],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of RAG predictions with comprehensive metrics

        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            retrieved_docs_list: List of retrieved document IDs for each query
            relevant_docs_list: List of relevant document IDs for each query
            k_values: List of K values for Recall@K and NDCG@K

        Returns:
            Dictionary with all evaluation metrics
        """
        metrics = defaultdict(list)

        # Retrieval metrics
        for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
            for k in k_values:
                recall = self.retrieval_metrics.recall_at_k(retrieved_docs, relevant_docs, k)
                ndcg = self.retrieval_metrics.ndcg_at_k(retrieved_docs, relevant_docs, k)
                metrics[f'recall@{k}'].append(recall)
                metrics[f'ndcg@{k}'].append(ndcg)

            mrr = self.retrieval_metrics.mean_reciprocal_rank(retrieved_docs, relevant_docs)
            metrics['mrr'].append(mrr)

        # Generation metrics - ROUGE
        for pred, ref in zip(predictions, references):
            rouge = self.generation_metrics.rouge_scores(pred, ref)
            for key, value in rouge.items():
                metrics[key].append(value)

            em = self.generation_metrics.exact_match(pred, ref)
            metrics['exact_match'].append(em)

        # Generation metrics - BERTScore (batch computation)
        if predictions and references:
            bert_scores = self.generation_metrics.bert_score(predictions, references)
            for key, value in bert_scores.items():
                metrics[key] = value  # BERTScore is already averaged

        # Average all metrics
        result = {}
        for key, values in metrics.items():
            if isinstance(values, list):
                result[key] = np.mean(values)
            else:
                result[key] = values

        return result

    def evaluate_single(
        self,
        prediction: str,
        reference: str,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG prediction

        Args:
            prediction: Generated answer
            reference: Ground truth answer
            retrieved_docs: Retrieved document IDs
            relevant_docs: Relevant document IDs
            k_values: List of K values for Recall@K and NDCG@K

        Returns:
            Dictionary with evaluation metrics
        """
        return self.evaluate_batch(
            predictions=[prediction],
            references=[reference],
            retrieved_docs_list=[retrieved_docs],
            relevant_docs_list=[relevant_docs],
            k_values=k_values
        )

    def evaluate_generation_only(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate only generation quality (no retrieval metrics)

        Use this when you only want to compare generated answers vs ground truth,
        without evaluating document retrieval performance.

        Args:
            predictions: List of generated answers
            references: List of ground truth answers

        Returns:
            Dictionary with generation metrics only (ROUGE, BERTScore, Exact Match)
        """
        metrics = defaultdict(list)

        # Generation metrics - ROUGE and Exact Match
        for pred, ref in zip(predictions, references):
            rouge = self.generation_metrics.rouge_scores(pred, ref)
            for key, value in rouge.items():
                metrics[key].append(value)

            em = self.generation_metrics.exact_match(pred, ref)
            metrics['exact_match'].append(em)

        # Generation metrics - BERTScore (batch computation)
        if predictions and references:
            bert_scores = self.generation_metrics.bert_score(predictions, references)
            for key, value in bert_scores.items():
                metrics[key] = value  # BERTScore is already averaged

        # Average all metrics
        result = {}
        for key, values in metrics.items():
            if isinstance(values, list):
                result[key] = np.mean(values)
            else:
                result[key] = values

        return result


class BenchRAGEvaluator:
    """
    Bench-RAG Evaluation System

    Comprehensive evaluation system for RAG models following the Bench-RAG framework
    """

    def __init__(self, k_values: List[int] = [1, 3, 5]):
        self.rag_metrics = RAGMetrics()
        self.k_values = k_values

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        model_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset with Bench-RAG metrics

        Args:
            dataset: List of data samples with ground truth
                Each sample should have: 'question', 'answer', 'documents', 'correct_doc_id'
            model_predictions: List of model predictions
                Each prediction should have: 'answer', 'retrieved_doc_ids'

        Returns:
            Dictionary with comprehensive evaluation results
        """
        predictions = []
        references = []
        retrieved_docs_list = []
        relevant_docs_list = []

        for data, pred in zip(dataset, model_predictions):
            predictions.append(pred['answer'])
            references.append(data['answer'])
            retrieved_docs_list.append(pred.get('retrieved_doc_ids', []))

            # Get relevant doc IDs from data
            correct_doc_id = data.get('correct_doc_id')
            if correct_doc_id:
                relevant_docs_list.append([correct_doc_id])
            else:
                # Find correct documents from documents list
                relevant_docs = [
                    doc['doc_id'] for doc in data.get('documents', [])
                    if doc.get('is_correct', False)
                ]
                relevant_docs_list.append(relevant_docs)

        # Compute all metrics
        metrics = self.rag_metrics.evaluate_batch(
            predictions=predictions,
            references=references,
            retrieved_docs_list=retrieved_docs_list,
            relevant_docs_list=relevant_docs_list,
            k_values=self.k_values
        )

        # Add summary statistics
        metrics['num_samples'] = len(dataset)

        return metrics

    def format_results(self, metrics: Dict[str, float]) -> str:
        """Format evaluation results for display"""
        lines = ["=" * 60]
        lines.append("BENCH-RAG EVALUATION RESULTS")
        lines.append("=" * 60)

        # Retrieval metrics
        lines.append("\n📊 Retrieval Metrics:")
        for k in self.k_values:
            if f'recall@{k}' in metrics:
                lines.append(f"  Recall@{k:2d}:  {metrics[f'recall@{k}']:.4f}")
        for k in self.k_values:
            if f'ndcg@{k}' in metrics:
                lines.append(f"  NDCG@{k:2d}:    {metrics[f'ndcg@{k}']:.4f}")
        if 'mrr' in metrics:
            lines.append(f"  MRR:        {metrics['mrr']:.4f}")

        # Generation metrics
        lines.append("\n📝 Generation Metrics:")
        if 'rouge1' in metrics:
            lines.append(f"  ROUGE-1:    {metrics['rouge1']:.4f}")
        if 'rouge2' in metrics:
            lines.append(f"  ROUGE-2:    {metrics['rouge2']:.4f}")
        if 'rougeL' in metrics:
            lines.append(f"  ROUGE-L:    {metrics['rougeL']:.4f}")
        if 'bert_f1' in metrics:
            lines.append(f"  BERTScore:  {metrics['bert_f1']:.4f}")
        if 'exact_match' in metrics:
            lines.append(f"  Exact Match: {metrics['exact_match']:.4f}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def evaluate_generation_only(
        self,
        dataset: List[Dict[str, Any]],
        model_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate only generation quality (no retrieval metrics)

        Use this when you want to compare generated answers vs ground truth
        without considering document retrieval performance.

        Args:
            dataset: List of data samples with ground truth
                Each sample should have: 'question', 'answer'
            model_predictions: List of model predictions
                Each prediction should have: 'answer'

        Returns:
            Dictionary with generation metrics only
        """
        predictions = []
        references = []

        for data, pred in zip(dataset, model_predictions):
            predictions.append(pred['answer'])
            references.append(data['answer'])

        # Compute generation metrics only
        metrics = self.rag_metrics.evaluate_generation_only(
            predictions=predictions,
            references=references
        )

        # Add summary statistics
        metrics['num_samples'] = len(dataset)

        return metrics

    def format_results_generation_only(self, metrics: Dict[str, float]) -> str:
        """Format generation-only evaluation results for display"""
        lines = ["=" * 60]
        lines.append("GENERATION-ONLY EVALUATION RESULTS")
        lines.append("=" * 60)

        # Generation metrics
        lines.append("\n📝 Generation Metrics:")
        if 'rouge1' in metrics:
            lines.append(f"  ROUGE-1:     {metrics['rouge1']:.4f}")
        if 'rouge2' in metrics:
            lines.append(f"  ROUGE-2:     {metrics['rouge2']:.4f}")
        if 'rougeL' in metrics:
            lines.append(f"  ROUGE-L:     {metrics['rougeL']:.4f}")
        if 'bert_f1' in metrics:
            lines.append(f"  BERTScore F1: {metrics['bert_f1']:.4f}")
        if 'bert_precision' in metrics:
            lines.append(f"  BERTScore P:  {metrics['bert_precision']:.4f}")
        if 'bert_recall' in metrics:
            lines.append(f"  BERTScore R:  {metrics['bert_recall']:.4f}")
        if 'exact_match' in metrics:
            lines.append(f"  Exact Match:  {metrics['exact_match']:.4f}")

        if 'num_samples' in metrics:
            lines.append(f"\n📊 Total Samples: {metrics['num_samples']}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


# Convenience function for easy import
def create_evaluator(k_values: List[int] = [1, 3, 5]) -> BenchRAGEvaluator:
    """Create a Bench-RAG evaluator instance"""
    return BenchRAGEvaluator(k_values=k_values)
