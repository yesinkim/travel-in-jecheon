"""
Train/Test Split Script

This script splits the formatted training data into train and test sets
with stratified sampling to maintain question type distribution.

Input: data/processed/training_data.jsonl
Output:
  - data/processed/train.jsonl
  - data/processed/test.jsonl
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


class TrainTestSplitter:
    """Splits dataset into train and test sets with stratification."""

    def __init__(self, test_size: float = 0.21, random_seed: int = 42):
        """
        Initialize splitter.

        Args:
            test_size: Proportion of data for test set (default: 0.21 for 79/21 split)
            random_seed: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_seed = random_seed
        random.seed(random_seed)

    def load_data(self, input_path: str) -> List[Dict[str, Any]]:
        """Load formatted training data from JSONL file."""
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def stratified_split(
        self,
        data: List[Dict[str, Any]],
        stratify_key: str = "question_type"
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Perform stratified split to maintain distribution of question types.

        Args:
            data: Full dataset
            stratify_key: Key to stratify on (default: "question_type")

        Returns:
            Tuple of (train_data, test_data)
        """
        # Group data by stratification key
        grouped_data = defaultdict(list)
        for item in data:
            key_value = item.get(stratify_key, "unknown")
            grouped_data[key_value].append(item)

        train_data = []
        test_data = []

        # Split each group proportionally
        for key_value, items in grouped_data.items():
            # Shuffle items in this group
            random.shuffle(items)

            # Calculate split point
            n_test = max(1, int(len(items) * self.test_size))
            n_train = len(items) - n_test

            # Split
            train_data.extend(items[:n_train])
            test_data.extend(items[n_train:])

            print(f"  {key_value}: {n_train} train, {n_test} test")

        # Shuffle final datasets
        random.shuffle(train_data)
        random.shuffle(test_data)

        return train_data, test_data

    def save_to_jsonl(self, data: List[Dict[str, Any]], output_path: str):
        """Save data to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✅ Saved {len(data)} items to {output_path}")

    def print_summary(
        self,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]]
    ):
        """Print summary of train/test split."""
        total = len(train_data) + len(test_data)

        print(f"\n📊 Train/Test Split Summary:")
        print(f"Total items: {total}")
        print(f"Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
        print(f"Test: {len(test_data)} ({len(test_data)/total*100:.1f}%)")

        # Question type distribution in train
        train_type_counts = {}
        for item in train_data:
            q_type = item.get("question_type", "unknown")
            train_type_counts[q_type] = train_type_counts.get(q_type, 0) + 1

        print("\n📂 Train Set - Question Type Distribution:")
        for q_type, count in sorted(train_type_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(train_data)) * 100
            print(f"  - {q_type}: {count} ({percentage:.1f}%)")

        # Question type distribution in test
        test_type_counts = {}
        for item in test_data:
            q_type = item.get("question_type", "unknown")
            test_type_counts[q_type] = test_type_counts.get(q_type, 0) + 1

        print("\n📂 Test Set - Question Type Distribution:")
        for q_type, count in sorted(test_type_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(test_data)) * 100
            print(f"  - {q_type}: {count} ({percentage:.1f}%)")

        # Difficulty distribution in train
        train_diff_counts = {}
        for item in train_data:
            diff = item.get("difficulty", "unknown")
            train_diff_counts[diff] = train_diff_counts.get(diff, 0) + 1

        print("\n📊 Train Set - Difficulty Distribution:")
        for diff, count in sorted(train_diff_counts.items()):
            percentage = (count / len(train_data)) * 100
            print(f"  - {diff}: {count} ({percentage:.1f}%)")

        # Difficulty distribution in test
        test_diff_counts = {}
        for item in test_data:
            diff = item.get("difficulty", "unknown")
            test_diff_counts[diff] = test_diff_counts.get(diff, 0) + 1

        print("\n📊 Test Set - Difficulty Distribution:")
        for diff, count in sorted(test_diff_counts.items()):
            percentage = (count / len(test_data)) * 100
            print(f"  - {diff}: {count} ({percentage:.1f}%)")


def main():
    """Main execution function."""
    print("🚀 Starting Train/Test Split...")

    # Paths
    input_path = "/home/user/goodganglabs/data/processed/training_data.jsonl"
    train_output_path = "/home/user/goodganglabs/data/processed/train.jsonl"
    test_output_path = "/home/user/goodganglabs/data/processed/test.jsonl"

    # Initialize splitter
    splitter = TrainTestSplitter(test_size=0.21, random_seed=42)

    # Load data
    data = splitter.load_data(input_path)
    print(f"✅ Loaded {len(data)} items")

    # Perform stratified split
    print("\n🔄 Performing stratified split by question type...")
    train_data, test_data = splitter.stratified_split(data, stratify_key="question_type")

    # Save splits
    print("\n💾 Saving train and test sets...")
    splitter.save_to_jsonl(train_data, train_output_path)
    splitter.save_to_jsonl(test_data, test_output_path)

    # Print summary
    splitter.print_summary(train_data, test_data)

    print("\n✅ Train/test split completed!")


if __name__ == "__main__":
    main()
