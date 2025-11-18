"""
Q&A Generation Script using OpenAI gemini API

This script generates question-answer pairs from document chunks using OpenAI gemini API.
Generates diverse question types: factual, descriptive, recommendation, comparison, no-answer.

Input: data/chunks/documents.jsonl
Output: data/chunks/qa_pairs.jsonl
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
# from openai import OpenAI
import google.generativeai as genai
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class QAGenerator:
    """Generates Q&A pairs using OpenAI gemini API."""

    # Question type distribution (target percentages)
    QUESTION_DISTRIBUTION = {
        "factual": 0.40,  # 40%
        "descriptive": 0.30,  # 30%
        "recommendation": 0.15,  # 15%
        "comparison": 0.10,  # 10%
        "no_answer": 0.05,  # 5%
    }

    # Questions per chunk by category (target: ~140-150 Q&A for 8B model training)
    QUESTIONS_PER_CHUNK = {
        "tourism": 8,  # Major tourist sites (most important)
        "transportation": 8,
        "food": 8,
        "accommodation": 8,
        "activity": 8,
        "culture": 8,
        "course": 7,
        "benefit": 7,
        "general": 7,
    }

    def __init__(self, api_key: str = None):
        """Initialize QA generator with OpenAI API key."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

            # response = self.client.chat.completions.create(
            #     model="gemini",
            #     messages=[
            #         {"role": "system", "content": "당신은 한국 관광 정보 데이터셋 생성 전문가입니다. 주어진 문서를 바탕으로 고품질의 질문-답변 쌍을 JSON 형식으로 생성합니다."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     max_tokens=4096,
            #     temperature=0.7,
            #     response_format={"type": "json_object"}

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config={"temperature": 0.7, "max_output_tokens": 4096, "response_mime_type": "application/json"},
        )
        self.qa_pairs = []

    def load_documents(self, documents_path: str) -> List[Dict[str, Any]]:
        """Load document chunks from JSONL file."""
        documents = []
        with open(documents_path, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))
        return documents

    def create_qa_generation_prompt(
        self,
        doc_content: str,
        doc_title: str,
        doc_category: str,
        num_questions: int
    ) -> str:
        """Create prompt for Claude to generate Q&A pairs."""

        prompt = f"""당신은 한국 관광 정보 데이터셋 생성 전문가입니다. 주어진 문서를 바탕으로 고품질의 질문-답변 쌍을 JSON 형식으로 생성합니다.
        제천시 관광 정보를 바탕으로 RAG(Retrieval-Augmented Generation) 학습용 데이터셋을 만들어주세요.

주어진 문서에서 {num_questions}개의 질문-답변 쌍을 생성해주세요.

[문서 정보]
제목: {doc_title}
카테고리: {doc_category}

[문서 내용]
{doc_content}

[생성 규칙]
1. 질문 유형 분포:
   - 사실 질문 (factual): ~40% - 주소, 시간, 가격 등 명확한 사실
   - 설명 질문 (descriptive): ~30% - "어떤 곳인가요?", "무엇을 할 수 있나요?"
   - 추천 질문 (recommendation): ~15% - "추천해주세요", "어디가 좋나요?"
   - 비교 질문 (comparison): ~10% - "차이점은?", "어떤 것이 나은가요?"
   - 정보 없음 (no_answer): ~5% - 문서에 답이 없는 질문

2. 답변 작성 원칙:
   - 반드시 주어진 문서 내용만 사용
   - 문서에 없는 정보는 추측하지 말 것
   - 정보가 없으면 "제공된 관광 정보에는 [주제]에 대한 내용이 없습니다" 형식으로 답변

   **중요: 추천/설명 질문 답변 시**
   - 구체적인 내용을 포함해야 함 (위치, 코스, 장소명, 특징 등)
   - 단순히 할인이나 가격 정보만 언급하지 말 것
   - 예: "시티투어 추천해주세요" → 어떤 코스를 가는지, 어떤 장소를 방문하는지 설명
   - 예: "맛집 추천해주세요" → 어떤 음식점이 있는지, 어떤 메뉴가 있는지 설명

3. 질문 품질:
   - 자연스러운 한국어 구어체
   - 실제 관광객이 물어볼 법한 질문
   - 너무 쉽거나 너무 어렵지 않게
   - 다양한 표현 사용 (반복 피하기)

[출력 형식]
JSON 배열로 반환하되, 각 요소는 다음 형식:
{{
  "question": "질문 내용",
  "answer": "답변 내용",
  "question_type": "factual|descriptive|recommendation|comparison|no_answer",
  "difficulty": "easy|medium|hard"
}}

예시:
[
  {{
    "question": "의림지는 어디에 있나요?",
    "answer": "의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다.",
    "question_type": "factual",
    "difficulty": "easy"
  }},
  {{
    "question": "의림지는 어떤 곳인가요?",
    "answer": "의림지는 제천 10경 중 하나로, 고대 수리시설의 원형을 간직한 역사적 장소입니다. 삼한시대에 축조된 것으로 알려져 있으며 한국관광 100선에 선정된 인기 명소입니다.",
    "question_type": "descriptive",
    "difficulty": "easy"
  }},
  {{
    "question": "의림지 주차요금은 얼마인가요?",
    "answer": "죄송합니다. 제공된 관광 정보에는 의림지 주차요금에 대한 내용이 없습니다.",
    "question_type": "no_answer",
    "difficulty": "medium"
  }}
]

이제 위 문서를 바탕으로 {num_questions}개의 질문-답변 쌍을 JSON 형식으로 생성해주세요.

반드시 다음 형식으로 응답하세요:
{{
  "qa_pairs": [
    {{
      "question": "질문 내용",
      "answer": "답변 내용",
      "question_type": "factual|descriptive|recommendation|comparison|no_answer",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}"""

        return prompt

    def generate_qa_for_document(
        self,
        doc: Dict[str, Any],
        num_questions: int = 7
    ) -> List[Dict[str, Any]]:
        """Generate Q&A pairs for a single document using OpenAI GPT-4o."""

        prompt = self.create_qa_generation_prompt(
            doc_content=doc["content"],
            doc_title=doc["title"],
            doc_category=doc["category"],
            num_questions=num_questions
        )

        try:

            response = self.model.generate_content(prompt)

            # Parse JSON response
            response_data = json.loads(response.text)

            # Handle different possible JSON structures
            if "qa_pairs" in response_data:
                qa_list = response_data["qa_pairs"]
            elif "questions" in response_data:
                qa_list = response_data["questions"]
            elif isinstance(response_data, list):
                qa_list = response_data
            else:
                # Try to find the first list value
                for value in response_data.values():
                    if isinstance(value, list):
                        qa_list = value
                        break
                else:
                    qa_list = []

            # Add document metadata to each Q&A
            for qa in qa_list:
                qa["doc_id"] = doc["doc_id"]
                qa["doc_title"] = doc["title"]
                qa["doc_category"] = doc["category"]
                qa["doc_content"] = doc["content"]

            return qa_list

        except Exception as e:
            print(f"\n❌ Error generating Q&A for {doc['doc_id']}: {e}")
            return []

    def generate_all_qa(
        self,
        documents: List[Dict[str, Any]],
        target_total: int = 140
    ) -> List[Dict[str, Any]]:
        """Generate Q&A pairs for all documents."""

        print(f"\n🤖 Generating Q&A pairs using Gemini 2.5pro API...")
        print(f"Target: {target_total} Q&A pairs from {len(documents)} documents\n")

        all_qa = []

        for doc in tqdm(documents, desc="Generating Q&A"):
            # Determine number of questions for this document
            num_questions = self.QUESTIONS_PER_CHUNK.get(
                doc["category"],
                self.QUESTIONS_PER_CHUNK["general"]
            )

            # Generate Q&A pairs
            qa_list = self.generate_qa_for_document(doc, num_questions)

            all_qa.extend(qa_list)

            # Rate limiting (respect Claude API limits)
            time.sleep(1)

            # Show progress
            print(f"  [{doc['doc_id']}] {doc['title']}: Generated {len(qa_list)} Q&A pairs")

        self.qa_pairs = all_qa
        return all_qa

    def save_to_jsonl(self, output_path: str):
        """Save Q&A pairs to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in self.qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')

        print(f"\n✅ Saved {len(self.qa_pairs)} Q&A pairs to {output_path}")

    def print_summary(self):
        """Print summary of generated Q&A pairs."""
        print(f"\n📊 Q&A Generation Summary:")
        print(f"Total Q&A pairs: {len(self.qa_pairs)}")

        # Question type distribution
        type_counts = {}
        for qa in self.qa_pairs:
            q_type = qa.get("question_type", "unknown")
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        print("\n📂 Question Type Distribution:")
        for q_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(self.qa_pairs)) * 100
            print(f"  - {q_type}: {count} ({percentage:.1f}%)")

        # Category distribution
        category_counts = {}
        for qa in self.qa_pairs:
            cat = qa.get("doc_category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print("\n📂 Category Distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {count}")

        # Difficulty distribution
        difficulty_counts = {}
        for qa in self.qa_pairs:
            diff = qa.get("difficulty", "unknown")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        print("\n📂 Difficulty Distribution:")
        for diff, count in sorted(difficulty_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(self.qa_pairs)) * 100
            print(f"  - {diff}: {count} ({percentage:.1f}%)")

        print("\n📝 Sample Q&A Pairs:")
        for i, qa in enumerate(self.qa_pairs[:3], 1):
            print(f"\n  Example {i}:")
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer'][:100]}...")
            print(f"  Type: {qa['question_type']} | Difficulty: {qa['difficulty']}")


def main():
    """Main execution function."""
    print("🚀 Starting Q&A Generation with OpenAI gemini API...")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ Error: GOOGLE_API_KEY environment variable not set!")
        print("Please set it using: export GOOGLE_API_KEY='your-api-key'")
        return

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    documents_path = os.path.join(project_root, "data", "chunks", "documents.jsonl")
    output_path = os.path.join(project_root, "data", "chunks", "qa_pairs.jsonl")

    # Initialize generator
    generator = QAGenerator(api_key=api_key)

    # Load documents
    documents = generator.load_documents(documents_path)
    print(f"✅ Loaded {len(documents)} documents")

    # Generate Q&A pairs
    qa_pairs = generator.generate_all_qa(documents, target_total=140)

    # Save to JSONL
    generator.save_to_jsonl(output_path)

    # Print summary
    generator.print_summary()

    print("\n✅ Q&A generation completed!")


if __name__ == "__main__":
    main()
