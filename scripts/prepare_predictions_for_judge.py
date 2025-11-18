"""
Prepare Model Predictions for LLM-as-a-Judge Evaluation

Converts model inference results to the format required by LLM Judge:
{
    "filename": "제천시관광정보책자.pdf",
    "content": "[correct document content]",
    "question": "...",
    "response": "[model's answer]"
}

Usage:
    python scripts/prepare_predictions_for_judge.py \\
        --dataset data/processed/test_data.jsonl \\
        --predictions outputs/model_predictions.jsonl \\
        --output outputs/predictions_for_judge.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: str):
    """Load JSONL file"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, path: str):
    """Save to JSONL file"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def find_correct_document(documents):
    """Find the correct document from documents list"""
    for doc in documents:
        if doc.get('is_correct', False):
            return doc
    # If no correct doc marked, return first one
    return documents[0] if documents else None


def main():
    parser = argparse.ArgumentParser(description="Prepare predictions for LLM Judge evaluation")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to test dataset JSONL (contains questions, documents, answers)'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to model predictions JSONL (contains generated responses)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output JSONL file for LLM Judge'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='제천시관광정보책자.pdf',
        help='Source filename to include in output'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_jsonl(args.dataset)

    print(f"Loading predictions from {args.predictions}...")
    predictions = load_jsonl(args.predictions)

    if len(dataset) != len(predictions):
        print(f"ERROR: Dataset has {len(dataset)} samples but predictions has {len(predictions)}")
        return 1

    print(f"Processing {len(dataset)} samples...")

    # Prepare for judge
    judge_data = []

    for data_item, pred_item in zip(dataset, predictions):
        # Find correct document
        correct_doc = find_correct_document(data_item.get('documents', []))

        if not correct_doc:
            print(f"WARNING: No correct document found for question: {data_item['question'][:50]}...")
            content = ""
        else:
            content = correct_doc.get('content', '')

        # Extract generated response
        response = pred_item.get('generated_answer') or pred_item.get('answer') or pred_item.get('response', '')

        judge_item = {
            'filename': args.filename,
            'content': content,
            'question': data_item['question'],
            'response': response
        }

        judge_data.append(judge_item)

    # Save
    print(f"\nSaving {len(judge_data)} items to {args.output}...")
    save_jsonl(judge_data, args.output)

    print("\n✓ Preparation complete!")
    print(f"\nNext step: Run LLM Judge evaluation:")
    print(f"  python scripts/evaluate_with_llm_judge.py \\")
    print(f"    --predictions {args.output} \\")
    print(f"    --output {args.output.replace('.jsonl', '_results.jsonl')}")

    return 0


if __name__ == '__main__':
    exit(main())
