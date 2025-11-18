"""
LLM-as-a-Judge Evaluation Module (Bench-RAG Style)

Implements 4 evaluation metrics using LLM as judge:
- Accuracy: Whether response contains extra details not in content
- Helpfulness: How helpful the response is (1-10 scale)
- Relevance: How well response addresses the question (1-10 scale)
- Depth: Level of detail in the response (1-10 scale)
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import google.generativeai as genai
from tqdm import tqdm
import time
from dotenv import load_dotenv

load_dotenv()


@dataclass
class JudgmentResult:
    """Result of LLM judge evaluation"""
    accuracy: bool
    accuracy_explanation: str
    helpfulness: int
    helpfulness_explanation: str
    relevance: int
    relevance_explanation: str
    depth: int
    depth_explanation: str


class LLMJudge:
    """LLM-as-a-Judge evaluator for RAG responses (Bench-RAG style)"""

    # System messages for each metric
    ACCURACY_SYSTEM = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the accuracy of the response. You will check whether the response contains extra details not found in the piece of information provided. If extra details are found, accuracy is false. Otherwise, accuracy is true. Take note that if the response partially addresses the question, but did not provide extra details not found in the piece of information provided, the response will still be considered accurate (hence accuracy = true). Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the accuracy with true or false by strictly following this JSON format: { "accuracy_explanation": "<provide an explanation on accuracy, whether extra details outside the content were found.>", "accuracy": true/false }"""

    HELPFULNESS_SYSTEM = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the helpfulness of the response. You will check whether the AI assistant is helpful in answering the question based on the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the helpfulness on a scale of 1 to 10 by strictly following this JSON format: { "helpfulness_explanation": "<provide an explanation on helpfulness>", "helpfulness": <score> }"""

    RELEVANCE_SYSTEM = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the relevance of the response. You will check the relevance of the response by evaluating whether the response fully addresses the question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the relevance on a scale of 1 to 10 by strictly following this JSON format: { "relevance_explanation": "<provide an explanation on relevance>", "relevance": <score> }"""

    DEPTH_SYSTEM = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below, based solely on a piece of information extracted from a file provided below. Your evaluation should consider the depth of the response. You will check the depth of the response by evaluating the level of detail of the response in answering the question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the depth on a scale of 1 to 10 by strictly following this JSON format: { "depth_explanation": "<provide an explanation on depth>", "depth": <score> }"""

    @staticmethod
    def create_user_message(filename: str, content: str, question: str, response: str) -> str:
        """Create user message for evaluation (same for all metrics)"""
        return f"""[The Start of Provided Information Extracted from a File]
Filename: {filename}
Information: {content}
[The End of Provided Information]

[Question]
{question}

[The Start of Assistant's Response]
{response}
[The End of Assistant's Response]"""

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize LLM Judge with Gemini API"""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.generation_config = {
            "temperature": 0.3,  # Lower temperature for more consistent judgments
            "max_output_tokens": 1024,
            "response_mime_type": "application/json"
        }

    def _call_llm(self, system_prompt: str, user_message: str, retry_count: int = 3) -> Dict[str, Any]:
        """Call Gemini API with retry logic"""
        for attempt in range(retry_count):
            try:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=self.generation_config,
                    system_instruction=system_prompt
                )

                response = model.generate_content(user_message)
                result = json.loads(response.text)
                return result

            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"API call failed (attempt {attempt + 1}/{retry_count}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"API call failed after {retry_count} attempts: {e}")
                    raise

    def evaluate_single(
        self,
        filename: str,
        content: str,
        question: str,
        response: str
    ) -> JudgmentResult:
        """
        Evaluate a single response using all 4 metrics

        Args:
            filename: Source document filename
            content: Factual content from document
            question: User's question
            response: Model's generated answer

        Returns:
            JudgmentResult with all 4 evaluations
        """
        user_message = self.create_user_message(filename, content, question, response)

        # Evaluate each metric
        accuracy_result = self._call_llm(self.ACCURACY_SYSTEM, user_message)
        helpfulness_result = self._call_llm(self.HELPFULNESS_SYSTEM, user_message)
        relevance_result = self._call_llm(self.RELEVANCE_SYSTEM, user_message)
        depth_result = self._call_llm(self.DEPTH_SYSTEM, user_message)

        return JudgmentResult(
            accuracy=accuracy_result['accuracy'],
            accuracy_explanation=accuracy_result['accuracy_explanation'],
            helpfulness=helpfulness_result['helpfulness'],
            helpfulness_explanation=helpfulness_result['helpfulness_explanation'],
            relevance=relevance_result['relevance'],
            relevance_explanation=relevance_result['relevance_explanation'],
            depth=depth_result['depth'],
            depth_explanation=depth_result['depth_explanation']
        )

    def evaluate_batch(
        self,
        predictions: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate batch of predictions

        Args:
            predictions: List of dicts with keys: filename, content, question, response
            show_progress: Whether to show progress bar

        Returns:
            List of evaluation results with all metrics
        """
        results = []

        iterator = tqdm(predictions, desc="Evaluating with LLM Judge") if show_progress else predictions

        for pred in iterator:
            try:
                judgment = self.evaluate_single(
                    filename=pred['filename'],
                    content=pred['content'],
                    question=pred['question'],
                    response=pred['response']
                )

                result = {
                    'filename': pred['filename'],
                    'question': pred['question'],
                    'response': pred['response'],
                    'accuracy': judgment.accuracy,
                    'accuracy_explanation': judgment.accuracy_explanation,
                    'helpfulness': judgment.helpfulness,
                    'helpfulness_explanation': judgment.helpfulness_explanation,
                    'relevance': judgment.relevance,
                    'relevance_explanation': judgment.relevance_explanation,
                    'depth': judgment.depth,
                    'depth_explanation': judgment.depth_explanation
                }

                results.append(result)

                # Small delay to avoid rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"\nError evaluating question '{pred['question'][:50]}...': {e}")
                # Add failed result with default values
                results.append({
                    'filename': pred['filename'],
                    'question': pred['question'],
                    'response': pred['response'],
                    'accuracy': None,
                    'accuracy_explanation': f"Evaluation failed: {e}",
                    'helpfulness': None,
                    'helpfulness_explanation': f"Evaluation failed: {e}",
                    'relevance': None,
                    'relevance_explanation': f"Evaluation failed: {e}",
                    'depth': None,
                    'depth_explanation': f"Evaluation failed: {e}"
                })

        return results

    def compute_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute aggregate metrics from evaluation results

        Args:
            results: List of evaluation results from evaluate_batch

        Returns:
            Dictionary with average scores for each metric
        """
        # Filter out failed evaluations (where values are None)
        valid_results = [r for r in results if r['accuracy'] is not None]

        if not valid_results:
            return {
                'accuracy_rate': 0.0,
                'avg_helpfulness': 0.0,
                'avg_relevance': 0.0,
                'avg_depth': 0.0,
                'num_evaluated': 0,
                'num_failed': len(results)
            }

        accuracy_count = sum(1 for r in valid_results if r['accuracy'])
        total_helpfulness = sum(r['helpfulness'] for r in valid_results)
        total_relevance = sum(r['relevance'] for r in valid_results)
        total_depth = sum(r['depth'] for r in valid_results)

        num_valid = len(valid_results)

        return {
            'accuracy_rate': accuracy_count / num_valid,
            'avg_helpfulness': total_helpfulness / num_valid,
            'avg_relevance': total_relevance / num_valid,
            'avg_depth': total_depth / num_valid,
            'num_evaluated': num_valid,
            'num_failed': len(results) - num_valid
        }

    def format_results(self, aggregate_metrics: Dict[str, float]) -> str:
        """Format aggregate metrics for display"""
        lines = ["=" * 60]
        lines.append("LLM-AS-A-JUDGE EVALUATION RESULTS (Bench-RAG)")
        lines.append("=" * 60)

        lines.append(f"\n📊 Aggregate Metrics:")
        lines.append(f"  Accuracy Rate:     {aggregate_metrics['accuracy_rate']:.2%}")
        lines.append(f"  Avg Helpfulness:   {aggregate_metrics['avg_helpfulness']:.2f}/10")
        lines.append(f"  Avg Relevance:     {aggregate_metrics['avg_relevance']:.2f}/10")
        lines.append(f"  Avg Depth:         {aggregate_metrics['avg_depth']:.2f}/10")

        lines.append(f"\n📈 Evaluation Stats:")
        lines.append(f"  Successfully Evaluated: {aggregate_metrics['num_evaluated']}")
        lines.append(f"  Failed Evaluations:     {aggregate_metrics['num_failed']}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def create_judge(api_key: str = None, model_name: str = "gemini-2.0-flash-exp") -> LLMJudge:
    """Convenience function to create LLM Judge"""
    return LLMJudge(api_key=api_key, model_name=model_name)
