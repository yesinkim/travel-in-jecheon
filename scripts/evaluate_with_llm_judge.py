"""
Evaluate RAG Model Predictions with LLM-as-a-Judge (Bench-RAG Style)

This script evaluates model predictions using 4 Bench-RAG metrics:
- Accuracy: No hallucination (extra details)
- Helpfulness: How helpful (1-10)
- Relevance: How relevant (1-10)
- Depth: Level of detail (1-10)

Usage:
    python scripts/evaluate_with_llm_judge.py --predictions predictions.jsonl --output results.jsonl

Input Format (predictions.jsonl):
    {
        "filename": "제천시관광정보책자.pdf",
        "content": "의림지는 제천시 송학면...",
        "question": "의림지는 어디에 있나요?",
        "response": "의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다."
    }

Output Format (results.jsonl):
    {
        "filename": "...",
        "question": "...",
        "response": "...",
        "accuracy": true,
        "accuracy_explanation": "...",
        "helpfulness": 9,
        "helpfulness_explanation": "...",
        "relevance": 10,
        "relevance_explanation": "...",
        "depth": 8,
        "depth_explanation": "..."
    }
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.llm_judge import create_judge


def load_predictions(predictions_path: str):
    """Load predictions from JSONL file"""
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def save_results(results, output_path: str):
    """Save evaluation results to JSONL file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG predictions with LLM-as-a-Judge")
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output results JSONL file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.0-flash-exp',
        help='Gemini model to use for judging (default: gemini-2.0-flash-exp)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of predictions to evaluate (for testing)'
    )

    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(args.predictions)

    if args.limit:
        predictions = predictions[:args.limit]
        print(f"Limiting to first {args.limit} predictions for testing")

    print(f"Loaded {len(predictions)} predictions")

    # Validate prediction format
    required_keys = ['filename', 'content', 'question', 'response']
    for i, pred in enumerate(predictions):
        missing = [k for k in required_keys if k not in pred]
        if missing:
            print(f"ERROR: Prediction {i} missing keys: {missing}")
            print(f"Required keys: {required_keys}")
            return 1

    # Create LLM Judge
    print(f"\nInitializing LLM Judge (model: {args.model})...")
    judge = create_judge(model_name=args.model)

    # Evaluate
    print(f"\nEvaluating {len(predictions)} predictions...")
    print("This will make 4 API calls per prediction (Accuracy, Helpfulness, Relevance, Depth)")
    print(f"Estimated API calls: {len(predictions) * 4}")

    results = judge.evaluate_batch(predictions, show_progress=True)

    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")
    aggregate = judge.compute_aggregate_metrics(results)

    # Display results
    print("\n" + judge.format_results(aggregate))

    # Save results
    print(f"\nSaving detailed results to {args.output}...")
    save_results(results, args.output)

    # Save aggregate metrics
    aggregate_path = args.output.replace('.jsonl', '_aggregate.json')
    with open(aggregate_path, 'w', encoding='utf-8') as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(f"Saved aggregate metrics to {aggregate_path}")

    print("\n✓ Evaluation complete!")
    return 0


if __name__ == '__main__':
    exit(main())
