"""
Distractor Addition Script

This script adds distractor (incorrect/irrelevant) documents to each Q&A pair
following the Finetune-RAG methodology.

Input:
  - data/chunks/documents.jsonl
  - data/chunks/qa_pairs.jsonl

Output:
  - data/chunks/qa_with_distractors.jsonl
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


class DistractorAdder:
    """Adds distractor documents to Q&A pairs for RAG fine-tuning."""

    def __init__(self, num_distractors: int = 2):
        """
        Initialize distractor adder.

        Following Finetune-RAG standard: 1 correct + 2 distractors = 3 total documents

        Args:
            num_distractors: Number of distractor documents to add per Q&A (default: 2)
        """
        self.num_distractors = num_distractors
        self.documents = []
        self.qa_pairs = []
        self.docs_by_category = defaultdict(list)

    def load_documents(self, documents_paths: List[str]) -> List[Dict[str, Any]]:
        """Load document chunks from a list of JSONL files."""
        documents = []
        for path in documents_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    # Ensure content field is consistent
                    if "content" in doc and "text" not in doc:
                        doc["text"] = doc.pop("content")
                    documents.append(doc)
                    # Index by category for efficient retrieval
                    self.docs_by_category[doc["category"]].append(doc)

        self.documents = documents
        return documents

    def load_qa_pairs(self, qa_path: str) -> List[Dict[str, Any]]:
        """Load Q&A pairs from JSONL file."""
        qa_pairs = []
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa_pairs.append(json.loads(line))

        self.qa_pairs = qa_pairs
        return qa_pairs

    def select_distractors(
        self,
        correct_doc_id: str,
        doc_category: str, # Kept for signature consistency, but no longer used in logic
        num_distractors: int
    ) -> List[Dict[str, Any]]:
        """
        Select distractor documents for a Q&A pair by randomly choosing
        from all documents, excluding the correct document.
        """
        # Create a pool of all documents excluding the correct one
        available_distractors = [
            doc for doc in self.documents
            if doc["doc_id"] != correct_doc_id
        ]

        # If there are enough available distractors, select randomly
        if len(available_distractors) >= num_distractors:
            distractors = random.sample(available_distractors, num_distractors)
        else:
            # If not enough unique distractors, return all available unique ones
            distractors = available_distractors
            print(f"⚠️ Warning: Not enough unique distractors available for {correct_doc_id}. Selected {len(distractors)} instead of {num_distractors}.")

        return distractors

    def add_distractors_to_qa(self) -> List[Dict[str, Any]]:
        """Add distractor documents to all Q&A pairs."""
        print(f"\n🎯 Adding {self.num_distractors} random distractors per Q&A pair...")
        print(f"  - Total documents per Q&A: {self.num_distractors + 1} (1 correct + {self.num_distractors} distractors)\n")

        qa_with_distractors = []

        for qa in self.qa_pairs:
            # For no_answer type questions, we don't have a correct document
            if qa.get("question_type") == "no_answer":
                # All documents are distractors
                distractors = random.sample(
                    self.documents,
                    min(self.num_distractors + 1, len(self.documents))
                )

                qa_enhanced = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "question_type": qa["question_type"],
                    "difficulty": qa.get("difficulty", "medium"),
                    "correct_doc_id": None,
                    "correct_doc": None,
                    "distractor_docs": distractors,
                }
            else:
                # Select distractors
                distractors = self.select_distractors(
                    correct_doc_id=qa["doc_id"],
                    doc_category=qa["doc_category"], # Kept for signature consistency
                    num_distractors=self.num_distractors
                )

                # Find correct document
                correct_doc = next(
                    (doc for doc in self.documents if doc["doc_id"] == qa["doc_id"]),
                    None
                )

                qa_enhanced = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "question_type": qa["question_type"],
                    "difficulty": qa.get("difficulty", "medium"),
                    "correct_doc_id": qa["doc_id"],
                    "correct_doc": correct_doc,
                    "distractor_docs": distractors,
                }

            qa_with_distractors.append(qa_enhanced)

        return qa_with_distractors

    def save_to_jsonl(self, qa_list: List[Dict[str, Any]], output_path: str):
        """Save Q&A pairs with distractors to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_list:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')

        print(f"✅ Saved {len(qa_list)} Q&A pairs with distractors to {output_path}")

    def print_summary(self, qa_list: List[Dict[str, Any]]):
        """Print summary of Q&A pairs with distractors."""
        print(f"\n📊 Distractor Addition Summary:")
        print(f"Total Q&A pairs: {len(qa_list)}")

        # Count distractors
        total_distractors = sum(
            len(qa.get("distractor_docs", [])) for qa in qa_list
        )
        print(f"Total distractor documents: {total_distractors}")
        print(f"Average distractors per Q&A: {total_distractors / len(qa_list):.1f}")

        # Question type distribution
        type_counts = {}
        for qa in qa_list:
            q_type = qa.get("question_type", "unknown")
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        print("\n📂 Question Type Distribution:")
        for q_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(qa_list)) * 100
            print(f"  - {q_type}: {count} ({percentage:.1f}%)")

        # Sample Q&A with distractors
        print("\n📝 Sample Q&A with Distractors:")
        sample_qa = qa_list[0]
        print(f"\nQuestion: {sample_qa['question']}")
        print(f"Answer: {sample_qa['answer'][:100]}...")
        print(f"Type: {sample_qa['question_type']} | Difficulty: {sample_qa['difficulty']}")

        if sample_qa['correct_doc']:
            print(f"\n✅ Correct Document: [{sample_qa['correct_doc_id']}] {sample_qa['correct_doc']['title']}")

        print(f"\n❌ Distractor Documents:")
        for i, distractor in enumerate(sample_qa['distractor_docs'], 1):
            print(f"  {i}. [{distractor['doc_id']}] {distractor['title']} ({distractor['category']})")


def main():
    """Main execution function."""
    print("🚀 Starting Distractor Addition...")

    # Paths
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # List of document paths to load
    document_paths = [
        os.path.join(project_root, "data", "chunks", "documents.jsonl"),
        os.path.join(project_root, "data", "chunks", "external_documents.jsonl")
    ]
    
    qa_path = os.path.join(project_root, "data", "chunks", "qa_pairs.jsonl")
    output_path = os.path.join(project_root, "data", "chunks", "qa_with_distractors.jsonl")

    # Initialize distractor adder (Finetune-RAG standard: 2 distractors)
    adder = DistractorAdder(num_distractors=2)

    # Load documents and Q&A pairs
    documents = adder.load_documents(document_paths)
    print(f"✅ Loaded {len(documents)} documents from {len(document_paths)} files")

    qa_pairs = adder.load_qa_pairs(qa_path)
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs")

    # Add distractors
    qa_with_distractors = adder.add_distractors_to_qa()

    # Save to JSONL
    adder.save_to_jsonl(qa_with_distractors, output_path)

    # Print summary
    adder.print_summary(qa_with_distractors)

    print("\n✅ Distractor addition completed!")


if __name__ == "__main__":
    main()
