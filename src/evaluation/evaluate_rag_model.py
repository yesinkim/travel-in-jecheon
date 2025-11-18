"""
Evaluation Script for Finetune-RAG Models

Based on the paper: "Finetune-RAG: Fine-Tuning Language Models to Resist
Hallucination in Retrieval-Augmented Generation"

This script evaluates:
1. Factuality (answers based on context)
2. Hallucination resistance (refusing to answer when context lacks information)
3. Answer relevance
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import torch
from dataclasses import dataclass
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk, Dataset


@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question: str
    context: str
    expected_answer: str
    generated_answer: str
    answer_type: str  # "answerable" or "unanswerable"
    is_correct: bool
    is_hallucination: bool
    score: float


class FinetuneRAGEvaluator:
    """
    Evaluator for Finetune-RAG models
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        base_model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize evaluator

        Args:
            model_path: Path to fine-tuned model (or adapter if using PEFT)
            tokenizer_path: Path to tokenizer (if different from model_path)
            base_model_path: Path to base model (if using PEFT adapter)
            device: Device to run evaluation on
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.base_model_path = base_model_path
        self.device = device

        self.model = None
        self.tokenizer = None

        self.load_model()

    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading tokenizer from {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Check if this is a PEFT adapter
        adapter_config_path = Path(self.model_path) / "adapter_config.json"
        is_peft_adapter = adapter_config_path.exists()

        if is_peft_adapter and self.base_model_path:
            print(f"Loading base model from {self.base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            print(f"Loading PEFT adapter from {self.model_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                torch_dtype=torch.bfloat16,
            )
        else:
            print(f"Loading model from {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.model.eval()
        print("Model loaded successfully")

    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate answer for a given question and context

        Args:
            question: Question to answer
            context: Retrieved context
            system_prompt: System prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = (
                "당신은 제공된 문맥(context)을 바탕으로 질문에 답변하는 AI 어시스턴트입니다. "
                "문맥에 답변이 포함되어 있다면 정확하게 답변하고, "
                "문맥에 답변이 없다면 '제공된 정보에서 답변을 찾을 수 없습니다'라고 답변하세요."
            )

        # Format input using XML structure
        user_content = f"""<document>
<source>제천시 관광정보</source>
<context>
{context}
</context>
</document>

<question>{question}</question>

<answer>"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def evaluate_single_example(
        self,
        question: str,
        context: str,
        expected_answer: str,
        answer_type: str,
    ) -> EvaluationResult:
        """
        Evaluate a single example

        Args:
            question: Question
            context: Context
            expected_answer: Expected answer
            answer_type: "answerable" or "unanswerable"

        Returns:
            EvaluationResult
        """
        # Generate answer
        generated_answer = self.generate_answer(question, context)

        # Check for hallucination
        is_hallucination = False
        if answer_type == "unanswerable":
            # Check if model correctly refuses to answer
            refusal_phrases = [
                "제공된 정보에서",
                "답변을 찾을 수 없습니다",
                "정보가 없습니다",
                "알 수 없습니다",
                "Not in context",
                "cannot answer",
            ]
            correctly_refused = any(phrase in generated_answer for phrase in refusal_phrases)
            is_correct = correctly_refused
            is_hallucination = not correctly_refused
        else:
            # Simple check: does the answer contain key information?
            # In a production system, you'd use more sophisticated metrics
            is_correct = self.check_answer_correctness(
                generated_answer,
                expected_answer,
                context
            )
            is_hallucination = False

        # Compute score
        score = 1.0 if is_correct else 0.0

        return EvaluationResult(
            question=question,
            context=context,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            answer_type=answer_type,
            is_correct=is_correct,
            is_hallucination=is_hallucination,
            score=score,
        )

    def check_answer_correctness(
        self,
        generated_answer: str,
        expected_answer: str,
        context: str,
    ) -> bool:
        """
        Check if generated answer is correct

        This is a simple heuristic. In production, you'd use:
        - BERTScore
        - ROUGE
        - LLM-as-judge (GPT-4)
        - Human evaluation

        Args:
            generated_answer: Generated answer
            expected_answer: Expected answer
            context: Context

        Returns:
            True if correct, False otherwise
        """
        # Simple heuristic: check if key entities/phrases from expected answer
        # are in generated answer
        # This is very basic - use proper metrics in production

        # Extract key terms from expected answer (split by space)
        expected_terms = set(expected_answer.split())

        # Check overlap
        generated_terms = set(generated_answer.split())
        overlap = expected_terms & generated_terms

        # If more than 50% of expected terms are in generated answer, consider it correct
        overlap_ratio = len(overlap) / len(expected_terms) if expected_terms else 0

        return overlap_ratio > 0.5

    def evaluate_dataset(
        self,
        dataset: Dataset,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset

        Args:
            dataset: Dataset to evaluate
            output_path: Path to save results

        Returns:
            Dictionary with evaluation metrics
        """
        results = []

        print(f"\nEvaluating {len(dataset)} examples...")

        for example in tqdm(dataset):
            # Extract fields
            messages = example.get("messages", [])

            if not messages:
                continue

            # Parse messages to extract question, context, answer
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

            if not user_msg or not assistant_msg:
                continue

            # Parse XML structure
            user_content = user_msg["content"]
            expected_answer = assistant_msg["content"]

            # Extract question and context from XML
            # This is a simple parser - you might want to use xml.etree or lxml
            try:
                context = user_content.split("<context>")[1].split("</context>")[0].strip()
                question = user_content.split("<question>")[1].split("</question>")[0].strip()
            except:
                print(f"Failed to parse example: {user_content[:100]}...")
                continue

            # Determine answer type
            answer_type = "unanswerable" if "제공된 정보에서 답변을 찾을 수 없습니다" in expected_answer else "answerable"

            # Evaluate
            result = self.evaluate_single_example(
                question=question,
                context=context,
                expected_answer=expected_answer,
                answer_type=answer_type,
            )

            results.append(result)

        # Compute aggregate metrics
        metrics = self.compute_metrics(results)

        # Save results if output path provided
        if output_path:
            self.save_results(results, metrics, output_path)

        return metrics

    def compute_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Compute aggregate metrics from evaluation results

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with metrics
        """
        if not results:
            return {}

        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        hallucinations = sum(1 for r in results if r.is_hallucination)

        # Separate answerable and unanswerable
        answerable = [r for r in results if r.answer_type == "answerable"]
        unanswerable = [r for r in results if r.answer_type == "unanswerable"]

        answerable_correct = sum(1 for r in answerable if r.is_correct)
        unanswerable_correct = sum(1 for r in unanswerable if r.is_correct)

        metrics = {
            "total_examples": total,
            "accuracy": correct / total if total > 0 else 0,
            "hallucination_rate": hallucinations / total if total > 0 else 0,
            "answerable_examples": len(answerable),
            "answerable_accuracy": answerable_correct / len(answerable) if answerable else 0,
            "unanswerable_examples": len(unanswerable),
            "unanswerable_accuracy": unanswerable_correct / len(unanswerable) if unanswerable else 0,
            "refusal_rate": unanswerable_correct / len(unanswerable) if unanswerable else 0,
        }

        return metrics

    def save_results(
        self,
        results: List[EvaluationResult],
        metrics: Dict[str, Any],
        output_path: Path,
    ):
        """
        Save evaluation results to file

        Args:
            results: List of evaluation results
            metrics: Aggregate metrics
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to dictionaries
        results_dict = [
            {
                "question": r.question,
                "context": r.context[:200] + "..." if len(r.context) > 200 else r.context,
                "expected_answer": r.expected_answer,
                "generated_answer": r.generated_answer,
                "answer_type": r.answer_type,
                "is_correct": r.is_correct,
                "is_hallucination": r.is_hallucination,
                "score": r.score,
            }
            for r in results
        ]

        # Save to JSON
        output_data = {
            "metrics": metrics,
            "results": results_dict,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {output_path}")

    def print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in a readable format"""
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Hallucination Rate: {metrics['hallucination_rate']*100:.2f}%")
        print()
        print(f"Answerable Questions:")
        print(f"  Count: {metrics['answerable_examples']}")
        print(f"  Accuracy: {metrics['answerable_accuracy']*100:.2f}%")
        print()
        print(f"Unanswerable Questions:")
        print(f"  Count: {metrics['unanswerable_examples']}")
        print(f"  Accuracy (Refusal Rate): {metrics['unanswerable_accuracy']*100:.2f}%")
        print("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Finetune-RAG Model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model or adapter"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to base model (if using PEFT adapter)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = FinetuneRAGEvaluator(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        device=args.device,
    )

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # Use test split if available, otherwise use train split
    eval_dataset = dataset.get("test", dataset.get("train"))

    # Evaluate
    metrics = evaluator.evaluate_dataset(
        eval_dataset,
        output_path=Path(args.output_path),
    )

    # Print metrics
    evaluator.print_metrics(metrics)


if __name__ == "__main__":
    main()
