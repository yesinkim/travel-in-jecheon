"""
Training Data Formatter

This script formats Q&A pairs with distractors into the training format
compatible with Finetune-RAG methodology and Hugging Face datasets.

Input: data/chunks/qa_with_distractors.jsonl
Output:
  - data/processed/training_data.jsonl (Finetune-RAG format)
  - data/processed/dataset_hf.jsonl (Hugging Face upload format)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple


class TrainingDataFormatter:
    """Formats Q&A pairs for RAG fine-tuning."""

    def __init__(self):
        """Initialize formatter."""
        self.qa_data = []

    def load_qa_with_distractors(self, input_path: str) -> List[Dict[str, Any]]:
        """Load Q&A pairs with distractors from JSONL file."""
        qa_data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                qa_data.append(json.loads(line))

        self.qa_data = qa_data
        return qa_data

    def format_for_instruction_tuning(
        self,
        qa: Dict[str, Any],
        format_style: str = "xml"
    ) -> Dict[str, Any]:
        """
        Format Q&A pair for instruction tuning.

        Args:
            qa: Q&A pair with distractors
            format_style: "xml" or "baseline"
        """
        # Collect all documents (correct + distractors)
        all_docs = []

        if qa["correct_doc"]:
            all_docs.append({
                "doc_id": qa["correct_doc"]["doc_id"],
                "title": qa["correct_doc"]["title"],
                "content": qa["correct_doc"]["content"],
                "is_correct": True,
            })

        for distractor in qa["distractor_docs"]:
            all_docs.append({
                "doc_id": distractor["doc_id"],
                "title": distractor["title"],
                "content": distractor["content"],
                "is_correct": False,
            })

        # Shuffle documents to randomize position of correct doc
        random.shuffle(all_docs)

        # Format documents based on style
        if format_style == "xml":
            documents_text = self._format_documents_xml(all_docs)
        else:
            documents_text = self._format_documents_baseline(all_docs)

        # Create instruction
        instruction = "제천 관광 정보를 바탕으로 질문에 답하세요. 제공된 문서들 중 관련 있는 정보만 사용하여 정확하게 답변해주세요."

        # Create full prompt
        full_prompt = f"{instruction}\n\n{documents_text}\n\n질문: {qa['question']}"

        # Format for instruction tuning (ChatML / Alpaca style)
        formatted = {
            "instruction": instruction,
            "documents": documents_text,
            "question": qa["question"],
            "answer": qa["answer"],
            "full_prompt": full_prompt,
            "question_type": qa["question_type"],
            "difficulty": qa["difficulty"],
            "correct_doc_id": qa.get("correct_doc_id"),
        }

        return formatted

    def _format_documents_xml(self, docs: List[Dict[str, Any]]) -> str:
        """Format documents in XML style."""
        xml_parts = ["<Documents>"]

        for i, doc in enumerate(docs, 1):
            xml_parts.append(f"  <Document id=\"{doc['doc_id']}\">")
            xml_parts.append(f"    <Title>{doc['title']}</Title>")
            xml_parts.append(f"    <Content>{doc['content']}</Content>")
            xml_parts.append(f"  </Document>")

        xml_parts.append("</Documents>")

        return "\n".join(xml_parts)

    def _format_documents_baseline(self, docs: List[Dict[str, Any]]) -> str:
        """Format documents in baseline style."""
        parts = []

        for i, doc in enumerate(docs, 1):
            parts.append(f"문서 {i} (ID: {doc['doc_id']})")
            parts.append(f"제목: {doc['title']}")
            parts.append(f"내용: {doc['content']}")
            parts.append("")  # Empty line between documents

        return "\n".join(parts)

    def format_for_huggingface(self, qa: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format Q&A pair for Hugging Face dataset upload.
        Follows the Finetune-RAG dataset structure.
        """
        # Get correct document
        correct_doc = qa.get("correct_doc")

        # Get distractor documents (up to 2 for Finetune-RAG compatibility)
        distractors = qa["distractor_docs"][:2]

        hf_format = {
            "question": qa["question"],
            "answer": qa["answer"],
            "content": correct_doc["content"] if correct_doc else "",
            "filename": correct_doc["filename"] if correct_doc else "",
            "fictitious_content1": distractors[0]["content"] if len(distractors) > 0 else "",
            "fictitious_filename1": distractors[0]["filename"] if len(distractors) > 0 else "",
            "fictitious_content2": distractors[1]["content"] if len(distractors) > 1 else "",
            "fictitious_filename2": distractors[1]["filename"] if len(distractors) > 1 else "",
            "question_type": qa["question_type"],
            "difficulty": qa["difficulty"],
        }

        return hf_format

    def format_all_data(
        self,
        format_style: str = "xml"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Format all Q&A pairs.

        Returns:
            Tuple of (instruction_tuning_data, huggingface_data)
        """
        print(f"\n🔄 Formatting {len(self.qa_data)} Q&A pairs...")

        instruction_data = []
        hf_data = []

        for qa in self.qa_data:
            # Format for instruction tuning
            inst_formatted = self.format_for_instruction_tuning(qa, format_style)
            instruction_data.append(inst_formatted)

            # Format for Hugging Face
            hf_formatted = self.format_for_huggingface(qa)
            hf_data.append(hf_formatted)

        return instruction_data, hf_data

    def save_to_jsonl(
        self,
        data: List[Dict[str, Any]],
        output_path: str
    ):
        """Save formatted data to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✅ Saved {len(data)} items to {output_path}")

    def print_summary(
        self,
        instruction_data: List[Dict[str, Any]],
        hf_data: List[Dict[str, Any]]
    ):
        """Print summary of formatted data."""
        print(f"\n📊 Formatting Summary:")
        print(f"Total formatted items: {len(instruction_data)}")

        # Question type distribution
        type_counts = {}
        for item in instruction_data:
            q_type = item.get("question_type", "unknown")
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        print("\n📂 Question Type Distribution:")
        for q_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(instruction_data)) * 100
            print(f"  - {q_type}: {count} ({percentage:.1f}%)")

        # Sample formatted data
        print("\n📝 Sample Formatted Data (Instruction Tuning):")
        sample = instruction_data[0]
        print(f"\nInstruction: {sample['instruction']}")
        print(f"\nDocuments (first 200 chars): {sample['documents'][:200]}...")
        print(f"\nQuestion: {sample['question']}")
        print(f"Answer: {sample['answer'][:100]}...")

        print("\n📝 Sample Formatted Data (Hugging Face):")
        hf_sample = hf_data[0]
        print(f"\nQuestion: {hf_sample['question']}")
        print(f"Answer: {hf_sample['answer'][:100]}...")
        print(f"Filename: {hf_sample['filename']}")
        print(f"Fictitious Filename 1: {hf_sample['fictitious_filename1']}")
        print(f"Fictitious Filename 2: {hf_sample['fictitious_filename2']}")


def main():
    """Main execution function."""
    print("🚀 Starting Training Data Formatting...")

    # Paths
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "data", "chunks", "qa_with_distractors.jsonl")
    output_instruction_path = os.path.join(project_root, "data", "processed", "training_data.jsonl")
    output_hf_path = os.path.join(project_root, "data", "processed", "dataset_hf.jsonl")

    # Initialize formatter
    formatter = TrainingDataFormatter()

    # Load Q&A data
    qa_data = formatter.load_qa_with_distractors(input_path)
    print(f"✅ Loaded {len(qa_data)} Q&A pairs with distractors")

    # Format all data
    instruction_data, hf_data = formatter.format_all_data(format_style="xml")

    # Save formatted data
    formatter.save_to_jsonl(instruction_data, output_instruction_path)
    formatter.save_to_jsonl(hf_data, output_hf_path)

    # Print summary
    formatter.print_summary(instruction_data, hf_data)

    print("\n✅ Training data formatting completed!")


if __name__ == "__main__":
    main()
