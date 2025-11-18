"""
Finetune-RAG Dataset Preparation

Based on the paper: "Finetune-RAG: Fine-Tuning Language Models to Resist
Hallucination in Retrieval-Augmented Generation"

This script prepares training data in XML-structured format to teach the model:
1. Use retrieved context effectively
2. Avoid hallucination when context doesn't have the answer
3. Respond with "Not in context" when appropriate
"""

from typing import List, Dict, Any
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class AnswerType(Enum):
    """Types of answers in the dataset"""
    ANSWERABLE = "answerable"  # Answer is in the context
    UNANSWERABLE = "unanswerable"  # Question cannot be answered from context


@dataclass
class RAGExample:
    """Single RAG training example"""
    question: str
    context: str
    answer: str
    answer_type: AnswerType
    source_page: int = None
    metadata: Dict[str, Any] = None


class FinetuneRAGDatasetBuilder:
    """
    Build Finetune-RAG dataset with XML-structured format
    """

    def __init__(
        self,
        xml_format: bool = True,
        unanswerable_ratio: float = 0.15,
        system_prompt: str = None
    ):
        """
        Args:
            xml_format: Use XML-structured format (recommended by paper)
            unanswerable_ratio: Ratio of unanswerable questions (paper uses ~10-15%)
            system_prompt: Custom system prompt for the model
        """
        self.xml_format = xml_format
        self.unanswerable_ratio = unanswerable_ratio

        # Default system prompt based on Finetune-RAG paper
        self.system_prompt = system_prompt or (
            "당신은 제공된 문맥(context)을 바탕으로 질문에 답변하는 AI 어시스턴트입니다. "
            "문맥에 답변이 포함되어 있다면 정확하게 답변하고, "
            "문맥에 답변이 없다면 '제공된 정보에서 답변을 찾을 수 없습니다'라고 답변하세요."
        )

    def format_example_xml(self, example: RAGExample) -> str:
        """
        Format a single example in XML structure (Finetune-RAG paper format)

        Args:
            example: RAG example

        Returns:
            XML-formatted string
        """
        if example.answer_type == AnswerType.ANSWERABLE:
            formatted = f"""<document>
<source>제천시 관광정보</source>
<context>
{example.context}
</context>
</document>

<question>{example.question}</question>

<answer>{example.answer}</answer>"""
        else:
            # Unanswerable question format
            formatted = f"""<document>
<source>제천시 관광정보</source>
<context>
{example.context}
</context>
</document>

<question>{example.question}</question>

<answer>제공된 정보에서 답변을 찾을 수 없습니다.</answer>"""

        return formatted

    def format_example_plain(self, example: RAGExample) -> str:
        """
        Format a single example in plain text structure

        Args:
            example: RAG example

        Returns:
            Plain text formatted string
        """
        if example.answer_type == AnswerType.ANSWERABLE:
            formatted = f"""### 문맥
{example.context}

### 질문
{example.question}

### 답변
{example.answer}"""
        else:
            formatted = f"""### 문맥
{example.context}

### 질문
{example.question}

### 답변
제공된 정보에서 답변을 찾을 수 없습니다."""

        return formatted

    def create_training_example(
        self,
        example: RAGExample,
        tokenizer_chat_template: bool = True
    ) -> Dict[str, str]:
        """
        Create a training example in the format expected by the model

        Args:
            example: RAG example
            tokenizer_chat_template: Use chat template format

        Returns:
            Dictionary with 'messages' or 'text' field
        """
        # Format the content
        if self.xml_format:
            content = self.format_example_xml(example)
        else:
            content = self.format_example_plain(example)

        if tokenizer_chat_template:
            # Chat template format (for instruction-tuned models)
            return {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content.split("<answer>")[0].strip()},
                    {"role": "assistant", "content": content.split("<answer>")[1].strip().replace("</answer>", "").strip()}
                ]
            }
        else:
            # Plain text format
            return {"text": f"{self.system_prompt}\n\n{content}"}

    def build_dataset(
        self,
        examples: List[RAGExample],
        output_path: Path,
        split_ratio: Dict[str, float] = {"train": 0.78, "test": 0.22}
    ):
        """
        Build complete dataset and save to files

        Args:
            examples: List of RAG examples
            output_path: Output directory path
            split_ratio: Train/test split ratio
        """
        from datasets import Dataset, DatasetDict
        import random

        # Shuffle examples
        random.shuffle(examples)

        # Create training examples
        training_data = [
            self.create_training_example(ex)
            for ex in examples
        ]

        # Split into train/test
        split_idx = int(len(training_data) * split_ratio["train"])
        train_data = training_data[:split_idx]
        test_data = training_data[split_idx:]

        # Create HuggingFace datasets
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "test": Dataset.from_list(test_data)
        })

        # Save to disk
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_path))

        # Also save as JSON for inspection
        with open(output_path / "train.json", "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(output_path / "test.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        print(f"Dataset saved to {output_path}")
        print(f"Train examples: {len(train_data)}")
        print(f"Test examples: {len(test_data)}")
        print(f"Total examples: {len(training_data)}")

        # Print statistics
        answerable_count = sum(
            1 for ex in examples if ex.answer_type == AnswerType.ANSWERABLE
        )
        unanswerable_count = len(examples) - answerable_count
        print(f"\nDataset Statistics:")
        print(f"  Answerable: {answerable_count} ({answerable_count/len(examples)*100:.1f}%)")
        print(f"  Unanswerable: {unanswerable_count} ({unanswerable_count/len(examples)*100:.1f}%)")

        return dataset_dict

    def validate_dataset(self, dataset_path: Path):
        """
        Validate the created dataset

        Args:
            dataset_path: Path to the dataset directory
        """
        from datasets import load_from_disk

        dataset = load_from_disk(str(dataset_path))

        print(f"\n=== Dataset Validation ===")
        print(f"Splits: {list(dataset.keys())}")

        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()} Split:")
            print(f"  Examples: {len(split_data)}")
            print(f"  Columns: {split_data.column_names}")

            # Sample example
            if len(split_data) > 0:
                print(f"\n  Sample Example:")
                sample = split_data[0]
                if "messages" in sample:
                    for msg in sample["messages"]:
                        print(f"    [{msg['role']}]: {msg['content'][:100]}...")
                else:
                    print(f"    {sample['text'][:200]}...")


def create_sample_dataset():
    """
    Create a sample Finetune-RAG dataset for demonstration
    """
    # Sample examples from Jecheon Tourism data
    examples = [
        RAGExample(
            question="의림지는 어디에 위치해 있나요?",
            context="의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다. 삼한시대에 축조된 것으로 알려진 고대 수리시설로, 제천 10경 중 하나입니다.",
            answer="의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다.",
            answer_type=AnswerType.ANSWERABLE,
            source_page=5
        ),
        RAGExample(
            question="의림지의 역사적 배경은 무엇인가요?",
            context="의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다. 삼한시대에 축조된 것으로 알려진 고대 수리시설로, 제천 10경 중 하나입니다.",
            answer="의림지는 삼한시대에 축조된 것으로 알려진 고대 수리시설이며, 제천 10경 중 하나로 역사적 가치가 높은 장소입니다.",
            answer_type=AnswerType.ANSWERABLE,
            source_page=5
        ),
        RAGExample(
            question="의림지 입장료는 얼마인가요?",
            context="의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다. 삼한시대에 축조된 것으로 알려진 고대 수리시설로, 제천 10경 중 하나입니다.",
            answer="제공된 정보에서 답변을 찾을 수 없습니다.",
            answer_type=AnswerType.UNANSWERABLE,
            source_page=5
        ),
        RAGExample(
            question="청풍호반 케이블카는 어떤 관광지인가요?",
            context="청풍호반 케이블카는 청풍호 주변의 아름다운 경관을 한눈에 볼 수 있는 관광 명소입니다. 케이블카를 타고 올라가면 청풍호와 주변 산세를 조망할 수 있습니다.",
            answer="청풍호반 케이블카는 청풍호 주변의 아름다운 경관을 한눈에 볼 수 있는 관광 명소로, 케이블카를 타고 올라가면 청풍호와 주변 산세를 조망할 수 있습니다.",
            answer_type=AnswerType.ANSWERABLE,
            source_page=8
        ),
        RAGExample(
            question="청풍호반 케이블카의 운영 시간은 언제인가요?",
            context="청풍호반 케이블카는 청풍호 주변의 아름다운 경관을 한눈에 볼 수 있는 관광 명소입니다. 케이블카를 타고 올라가면 청풍호와 주변 산세를 조망할 수 있습니다.",
            answer="제공된 정보에서 답변을 찾을 수 없습니다.",
            answer_type=AnswerType.UNANSWERABLE,
            source_page=8
        ),
    ]

    # Build dataset
    builder = FinetuneRAGDatasetBuilder(
        xml_format=True,
        unanswerable_ratio=0.15
    )

    output_path = Path("data/processed/finetune_rag_sample")
    dataset = builder.build_dataset(examples, output_path)

    # Validate
    builder.validate_dataset(output_path)

    return dataset


if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample Finetune-RAG dataset...")
    create_sample_dataset()
    print("\nDone! Dataset created at data/processed/finetune_rag_sample")
