"""
Train/Validation/Test Split Script

This script splits the formatted training data into train, validation, and test sets
with stratified sampling to maintain question type distribution.

Input: data/processed/training_data.jsonl
Output:
  - data/processed/train.jsonl
  - data/processed/val.jsonl
  - data/processed/test.jsonl
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class TrainTestSplitter:
    """Splits dataset into train, validation, and test sets with stratification."""

    def __init__(self, val_size: float = 0.1, test_size: float = 0.1, random_seed: int = 42):
        """
        Initialize splitter.

        Args:
            val_size: Proportion of data for validation set (default: 0.1 for 80/10/10 split)
            test_size: Proportion of data for test set (default: 0.1 for 80/10/10 split)
            random_seed: Random seed for reproducibility
        """
        self.val_size = val_size
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
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Perform stratified split to maintain distribution of question types.

        Args:
            data: Full dataset
            stratify_key: Key to stratify on (default: "question_type")

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Group data by stratification key
        grouped_data = defaultdict(list)
        for item in data:
            key_value = item.get(stratify_key, "unknown")
            grouped_data[key_value].append(item)

        train_data, val_data, test_data = [], [], []

        # Split each group proportionally
        for key_value, items in grouped_data.items():
            random.shuffle(items)

            n_total = len(items)
            n_val = int(n_total * self.val_size)
            n_test = int(n_total * self.test_size)
            # Handle cases where a group is too small
            if n_total > 2:
                if n_val == 0: n_val = 1
                if n_test == 0: n_test = 1
            elif n_total == 2:
                n_val = 1
                n_test = 1
            elif n_total == 1:
                n_val = 0
                n_test = 1 # Prioritize test set

            n_train = n_total - n_val - n_test
            if n_train < 0: n_train = 0


            train_data.extend(items[:n_train])
            val_data.extend(items[n_train:n_train + n_val])
            test_data.extend(items[n_train + n_val:])

            print(f"  {key_value}: {n_train} train, {n_val} val, {n_test} test")

        # Shuffle final datasets
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        return train_data, val_data, test_data

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
        val_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]]
    ):
        """Print summary of train/val/test split."""
        total = len(train_data) + len(val_data) + len(test_data)
        if total == 0:
            print("\n📊 No data to split.")
            return

        print(f"\n📊 Train/Val/Test Split Summary:")
        print(f"Total items: {total}")
        print(f"Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
        print(f"Validation: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
        print(f"Test: {len(test_data)} ({len(test_data)/total*100:.1f}%)")

        def print_distribution(dataset, name, key):
            counts = defaultdict(int)
            if not dataset: return
            for item in dataset:
                value = item.get(key, "unknown")
                counts[value] += 1

            print(f"\n📂 {name} Set - {key.replace('_', ' ').title()} Distribution:")
            for value, count in sorted(counts.items(), key=lambda x: -x[1]):
                percentage = (count / len(dataset)) * 100
                print(f"  - {value}: {count} ({percentage:.1f}%)")
        
        print_distribution(train_data, "Train", "question_type")
        print_distribution(val_data, "Validation", "question_type")
        print_distribution(test_data, "Test", "question_type")

        print_distribution(train_data, "Train", "difficulty")
        print_distribution(val_data, "Validation", "difficulty")
        print_distribution(test_data, "Test", "difficulty")


def main():
    """Main execution function."""
    print("🚀 Starting Train/Val/Test Split...")

    # Paths
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "data", "processed", "training_data.jsonl")
    train_output_path = os.path.join(project_root, "data", "processed", "train.jsonl")
    val_output_path = os.path.join(project_root, "data", "processed", "val.jsonl")
    test_output_path = os.path.join(project_root, "data", "processed", "test.jsonl")

    # Initialize splitter for 80/10/10 split
    splitter = TrainTestSplitter(val_size=0.1, test_size=0.1, random_seed=42)

    # Load data
    try:
        data = splitter.load_data(input_path)
        print(f"✅ Loaded {len(data)} items from {input_path}")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_path}")
        return

    # Perform stratified split
    print("\n🔄 Performing stratified split by question type...")
    train_data, val_data, test_data = splitter.stratified_split(data, stratify_key="question_type")

    # Save splits
    print("\n💾 Saving train, validation, and test sets...")
    splitter.save_to_jsonl(train_data, train_output_path)
    splitter.save_to_jsonl(val_data, val_output_path)
    splitter.save_to_jsonl(test_data, test_output_path)

    # Print summary
    splitter.print_summary(train_data, val_data, test_data)

    print("\n✅ Train/val/test split completed!")


if __name__ == "__main__":
    main()
