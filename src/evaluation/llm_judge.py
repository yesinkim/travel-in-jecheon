"""
LLM-based RAG Evaluation using Gemini 2.5 Pro

This module implements evaluation metrics for RAG systems using an LLM judge
following the evaluation framework described in the RAFT paper.

Metrics:
- Accuracy (true/false): Whether response contains extra details not found in source
- Helpfulness (1-10): How helpful the response is in answering the question
- Relevance (1-10): How well the response addresses the question
- Depth (1-10): Level of detail in the response
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)


class LLMJudge:
    """LLM-based judge for evaluating RAG system responses"""

    # Evaluation prompts from the paper
    SYSTEM_PROMPTS = {
        "accuracy": """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the accuracy of the response.

You will check whether the response contains extra details not found in the piece of information provided. If extra details are found, accuracy is false. Otherwise, accuracy is true. Take note that if the response partially addresses the question, but did not provide extra details not found in the piece of information provided, the response will still be considered accurate (hence accuracy = true).

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the accuracy with true or false by strictly following this JSON format:
{
  "accuracy_explanation": "<provide an explanation on accuracy, whether extra details outside the content were found.>",
  "accuracy": <true/false>
}""",

        "helpfulness": """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the helpfulness of the response.

You will check whether the AI assistant is helpful in answering the question based on the response.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the helpfulness on a scale of 1 to 10 by strictly following this JSON format:
{
  "helpfulness_explanation": "<provide an explanation on helpfulness>",
  "helpfulness": <score>
}""",

        "relevance": """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the relevance of the response.

You will check the relevance of the response by evaluating whether the response fully addresses the question.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the relevance on a scale of 1 to 10 by strictly following this JSON format:
{
  "relevance_explanation": "<provide an explanation on relevance>",
  "relevance": <score>
}""",

        "depth": """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the depth of the response.

You will check the depth of the response by evaluating the level of detail of the response in answering the question.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the depth on a scale of 1 to 10 by strictly following this JSON format:
{
  "depth_explanation": "<provide an explanation on depth>",
  "depth": <score>
}"""
    }

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        """
        Initialize LLM Judge with Gemini model

        Args:
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
        """
        self.model = genai.GenerativeModel(model_name)
        self.documents_cache = {}  # Cache for loaded documents

    def load_documents(self, documents_path: str) -> Dict[str, Dict]:
        """
        Load documents from JSONL file

        Args:
            documents_path: Path to documents.jsonl file

        Returns:
            Dictionary mapping doc_id to document data
        """
        if self.documents_cache:
            return self.documents_cache

        documents = {}
        with open(documents_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                documents[doc['doc_id']] = doc

        self.documents_cache = documents
        return documents

    def format_user_message(
        self,
        filename: str,
        content: str,
        question: str,
        response: str
    ) -> str:
        """
        Format user message following the paper's template

        Args:
            filename: Source document filename
            content: Retrieved document content
            question: User question
            response: Model response to evaluate

        Returns:
            Formatted user message
        """
        return f"""[The Start of Provided Information Extracted from a File]
Filename: {filename}
Information: {content}
[The End of Provided Information]

[Question]
{question}

[The Start of Assistant's Response]
{response}
[The End of Assistant's Response]"""

    def evaluate_metric(
        self,
        metric: str,
        filename: str,
        content: str,
        question: str,
        response: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate a single metric using LLM judge

        Args:
            metric: One of 'accuracy', 'helpfulness', 'relevance', 'depth'
            filename: Source document filename
            content: Retrieved document content
            question: User question
            response: Model response to evaluate
            max_retries: Maximum number of retries on API errors

        Returns:
            Dictionary with evaluation results
        """
        system_prompt = self.SYSTEM_PROMPTS[metric]
        user_message = self.format_user_message(filename, content, question, response)

        # Combine system prompt and user message for Gemini
        full_prompt = f"{system_prompt}\n\n{user_message}"

        for attempt in range(max_retries):
            try:
                # Call Gemini API
                result = self.model.generate_content(full_prompt)
                response_text = result.text

                # Extract JSON from response
                # The response should contain JSON, extract it
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1

                if json_start == -1 or json_end == 0:
                    raise ValueError(f"No JSON found in response: {response_text}")

                json_str = response_text[json_start:json_end]
                evaluation = json.loads(json_str)

                return evaluation

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Error evaluating {metric}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to evaluate {metric} after {max_retries} attempts: {e}")
                    # Return default error response
                    return {
                        f"{metric}_explanation": f"Evaluation failed: {str(e)}",
                        metric: None
                    }

    def evaluate_response(
        self,
        question: str,
        response: str,
        retrieved_docs: List[Dict],
        documents: Dict[str, Dict],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response across all metrics

        Args:
            question: User question
            response: Model response to evaluate
            retrieved_docs: List of retrieved document metadata
            documents: Dictionary of all documents
            metrics: List of metrics to evaluate (default: all)

        Returns:
            Dictionary with all evaluation results
        """
        if metrics is None:
            metrics = ["accuracy", "helpfulness", "relevance", "depth"]

        # Combine all retrieved document contents
        combined_content = []
        filenames = []

        for doc_meta in retrieved_docs:
            doc_id = doc_meta['doc_id']
            if doc_id in documents:
                doc = documents[doc_id]
                combined_content.append(f"[{doc['title']}]\n{doc['content']}")
                filenames.append(doc['filename'])

        content = "\n\n".join(combined_content)
        filename = ", ".join(filenames)

        # Evaluate each metric
        results = {}
        for metric in metrics:
            print(f"  Evaluating {metric}...")
            metric_result = self.evaluate_metric(
                metric=metric,
                filename=filename,
                content=content,
                question=question,
                response=response
            )
            results.update(metric_result)

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        return results


def evaluate_rag_results(results_path: str, documents_path: str, output_path: str, model_name: str = "gemini-2.5-pro") -> Dict[str, Any]:
    """
    Evaluate RAG comparison results using LLM judge

    Args:
        results_path: Path to rag_comparison.json
        documents_path: Path to documents.jsonl
        output_path: Path to save evaluation results
        model_name: Gemini model to use

    Returns:
        Dictionary with evaluation results
    """
    # Initialize judge
    judge = LLMJudge(model_name=model_name)

    # Load documents
    print("Loading documents...")
    documents = judge.load_documents(documents_path)
    print(f"Loaded {len(documents)} documents")

    # Load RAG results
    print(f"\nLoading RAG results from {results_path}...")
    with open(results_path, 'r', encoding='utf-8') as f:
        rag_results = json.load(f)
    print(f"Loaded {len(rag_results)} test cases")

    # Evaluate each result
    evaluated_results = []

    for i, result in enumerate(rag_results, 1):
        print(f"\n{'='*80}")
        print(f"Evaluating Test Case {i}/{len(rag_results)}")
        print(f"Question: {result['question']}")
        print(f"{'='*80}")

        evaluated_result = {
            "question": result["question"],
            "timestamp": result["timestamp"],
            "models": {}
        }

        for model_name, model_result in result["models"].items():
            print(f"\n🤖 Evaluating {model_name}...")

            # Evaluate response
            evaluation = judge.evaluate_response(
                question=result["question"],
                response=model_result["answer"],
                retrieved_docs=model_result["retrieved_docs"],
                documents=documents
            )

            evaluated_result["models"][model_name] = {
                "answer": model_result["answer"],
                "retrieved_docs": model_result["retrieved_docs"],
                "evaluation": evaluation
            }

            # Print results
            print(f"  ✓ Accuracy: {evaluation.get('accuracy', 'N/A')}")
            print(f"  ✓ Helpfulness: {evaluation.get('helpfulness', 'N/A')}/10")
            print(f"  ✓ Relevance: {evaluation.get('relevance', 'N/A')}/10")
            print(f"  ✓ Depth: {evaluation.get('depth', 'N/A')}/10")

        evaluated_results.append(evaluated_result)

    # Save results
    print(f"\n{'='*80}")
    print(f"Saving evaluation results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluated_results, f, ensure_ascii=False, indent=2)

    print(f"✓ Evaluation complete!")

    # Calculate aggregate statistics
    stats = calculate_aggregate_stats(evaluated_results)

    # Save statistics
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ Statistics saved to {stats_path}")

    return evaluated_results


def calculate_aggregate_stats(evaluated_results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate aggregate statistics across all evaluations

    Args:
        evaluated_results: List of evaluated results

    Returns:
        Dictionary with aggregate statistics
    """
    stats = {}

    # Get all model names
    model_names = list(evaluated_results[0]["models"].keys())

    for model_name in model_names:
        model_stats = {
            "accuracy_true_count": 0,
            "accuracy_false_count": 0,
            "accuracy_rate": 0.0,
            "helpfulness_scores": [],
            "relevance_scores": [],
            "depth_scores": [],
            "helpfulness_avg": 0.0,
            "relevance_avg": 0.0,
            "depth_avg": 0.0
        }

        for result in evaluated_results:
            eval_data = result["models"][model_name]["evaluation"]

            # Accuracy (boolean)
            if eval_data.get("accuracy") is True:
                model_stats["accuracy_true_count"] += 1
            elif eval_data.get("accuracy") is False:
                model_stats["accuracy_false_count"] += 1

            # Numeric metrics
            if eval_data.get("helpfulness") is not None:
                model_stats["helpfulness_scores"].append(eval_data["helpfulness"])

            if eval_data.get("relevance") is not None:
                model_stats["relevance_scores"].append(eval_data["relevance"])

            if eval_data.get("depth") is not None:
                model_stats["depth_scores"].append(eval_data["depth"])

        # Calculate averages
        total_accuracy = model_stats["accuracy_true_count"] + model_stats["accuracy_false_count"]
        if total_accuracy > 0:
            model_stats["accuracy_rate"] = model_stats["accuracy_true_count"] / total_accuracy

        if model_stats["helpfulness_scores"]:
            model_stats["helpfulness_avg"] = sum(model_stats["helpfulness_scores"]) / len(model_stats["helpfulness_scores"])

        if model_stats["relevance_scores"]:
            model_stats["relevance_avg"] = sum(model_stats["relevance_scores"]) / len(model_stats["relevance_scores"])

        if model_stats["depth_scores"]:
            model_stats["depth_avg"] = sum(model_stats["depth_scores"]) / len(model_stats["depth_scores"])

        stats[model_name] = model_stats

    return stats


if __name__ == "__main__":
    # Default paths
    RESULTS_PATH = "notebooks/results/rag_comparison.json"
    DOCUMENTS_PATH = "data/chunks/documents.jsonl"
    OUTPUT_PATH = "results/evaluation/rag_evaluation_gemini.json"

    # Run evaluation
    evaluate_rag_results(
        results_path=RESULTS_PATH,
        documents_path=DOCUMENTS_PATH,
        output_path=OUTPUT_PATH,
        model_name="gemini-2.5-pro"
    )
