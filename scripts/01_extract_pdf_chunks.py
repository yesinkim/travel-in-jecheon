"""
Document Chunking Script for Jecheon Tourism Dataset

This script extracts meaningful chunks from the Jecheon tourism markdown file
and creates a structured documents.jsonl file.

Output: data/chunks/documents.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


class JecheonDocumentChunker:
    """Extracts and chunks Jecheon tourism information."""

    CATEGORIES = {
        "transportation": ["시티투어", "관광택시", "관광주민증"],
        "tourism": ["의림지", "청풍", "배론", "박달재", "옥순봉", "케이블카"],
        "food": ["맛집", "식당", "음식"],
        "accommodation": ["숙박", "리조트", "호텔", "게스트하우스"],
        "activity": ["트레킹", "체험", "축제", "레저"],
        "culture": ["박물관", "성지", "문화", "역사"],
        "course": ["코스", "여행", "추천"],
        "benefit": ["인센티브", "할인", "혜택", "기부"],
    }

    def __init__(self, markdown_path: str):
        """Initialize chunker with markdown file path."""
        self.markdown_path = Path(markdown_path)
        self.chunks = []

    def read_markdown(self) -> str:
        """Read markdown file."""
        with open(self.markdown_path, 'r', encoding='utf-8') as f:
            return f.read()

    def clean_text(self, text: str) -> str:
        """Clean text by removing line numbers and extra whitespace."""
        # Remove line numbers (e.g., "    1→")
        text = re.sub(r'^\s*\d+→', '', text, flags=re.MULTILINE)
        # Remove multiple blank lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def categorize_chunk(self, content: str, title: str) -> str:
        """Determine the category of a chunk based on its content and title."""
        text = (title + " " + content).lower()

        category_scores = {}
        for category, keywords in self.CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            if score > 0:
                category_scores[category] = score

        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return "general"

    def extract_chunks(self) -> List[Dict[str, Any]]:
        """Extract meaningful chunks from the markdown content."""
        content = self.read_markdown()
        content = self.clean_text(content)

        # Split by major sections
        sections = []

        # Manual parsing based on content structure
        chunks_data = [
            # Transportation & Services
            {
                "title": "디지털관광주민증",
                "category": "transportation",
                "content": self._extract_section(content, "디지털 관광주민증", "제천 시티투어"),
                "page": 4,
            },
            {
                "title": "제천 시티투어",
                "category": "transportation",
                "content": self._extract_section(content, "제천 시티투어", "제천 관광택시"),
                "page": 5,
            },
            {
                "title": "제천 관광택시",
                "category": "transportation",
                "content": self._extract_section(content, "제천 관광택시", "단체관광객 유치 인센티브"),
                "page": 6,
            },
            {
                "title": "단체관광객 인센티브",
                "category": "benefit",
                "content": self._extract_section(content, "단체관광객 유치 인센티브", "가스트로 투어"),
                "page": 7,
            },
            {
                "title": "가스트로 투어",
                "category": "food",
                "content": self._extract_section(content, "가스트로 투어", "모바일 바로가기"),
                "page": 8,
            },

            # Tourism Sites - Major
            {
                "title": "의림지·의림지역사박물관",
                "category": "tourism",
                "content": self._extract_section(content, "의림지·의림지역사박물관", "배론성지"),
                "page": 12,
                "location": "송학면",
                "address": "제천시 송학면 의림대로 47길 7",
            },
            {
                "title": "배론성지",
                "category": "culture",
                "content": self._extract_section(content, "배론성지", "박달재"),
                "page": 12,
                "location": "봉양읍",
                "address": "제천시 봉양읍 배론성지길 296",
            },
            {
                "title": "박달재",
                "category": "tourism",
                "content": self._extract_section(content, "박달재", "제천한방엑스포 공원"),
                "page": 12,
                "location": "백운면",
                "address": "제천시 백운면 박달로 231",
            },
            {
                "title": "제천한방엑스포 공원",
                "category": "culture",
                "content": self._extract_section(content, "제천한방엑스포 공원", "의림지 수리공원"),
                "page": 12,
                "address": "제천시 한방엑스포로 19",
            },
            {
                "title": "청풍호반 케이블카",
                "category": "tourism",
                "content": self._extract_section(content, "청풍호반 케이블카", "청풍문화유산단지"),
                "page": 14,
                "location": "청풍면",
                "address": "제천시 청풍면 문화재길 166",
            },
            {
                "title": "청풍문화유산단지",
                "category": "culture",
                "content": self._extract_section(content, "청풍문화유산단지", "청풍랜드"),
                "page": 14,
                "location": "청풍면",
                "address": "제천시 청풍호로 2048",
            },
            {
                "title": "청풍랜드",
                "category": "activity",
                "content": self._extract_section(content, "청풍랜드", "청풍호 자드락길"),
                "page": 14,
                "location": "청풍면",
                "address": "제천시 청풍면 청풍호로50길 6",
            },
            {
                "title": "옥순봉 출렁다리",
                "category": "tourism",
                "content": self._extract_section(content, "옥순봉 출렁다리", "충주호 크루즈"),
                "page": 14,
                "location": "수산면",
                "address": "제천시 수산면 옥순봉로342",
            },

            # Activities & Experiences
            {
                "title": "트레킹·걷기 좋은 곳",
                "category": "activity",
                "content": self._extract_section(content, "트래킹·걷기 좋은곳", "코스여행 추천"),
                "page": 16,
            },

            # Travel Courses
            {
                "title": "1일 코스",
                "category": "course",
                "content": self._extract_section(content, "1일 코스", "1박 2일 코스"),
                "page": 17,
            },
            {
                "title": "1박 2일 코스",
                "category": "course",
                "content": self._extract_section(content, "1박 2일 코스", "휴양·힐링 코스"),
                "page": 17,
            },
            {
                "title": "슬로시티 코스",
                "category": "course",
                "content": self._extract_section(content, "#슬로시티 코스", "#북부·서부권 코스"),
                "page": 17,
            },
            {
                "title": "북부·서부권 코스",
                "category": "course",
                "content": self._extract_section(content, "#북부·서부권 코스", "문화·역사 코스"),
                "page": 17,
            },

            # Food
            {
                "title": "제천맛집 소개",
                "category": "food",
                "content": self._extract_section(content, "제천맛집", "주요 숙박시설"),
                "page": 20,
            },

            # Accommodation
            {
                "title": "주요 숙박시설",
                "category": "accommodation",
                "content": self._extract_section(content, "주요 숙박시설", "고향사랑 기부제"),
                "page": 22,
            },

            # Benefits & Tips
            {
                "title": "고향사랑 기부제",
                "category": "benefit",
                "content": self._extract_section(content, "고향사랑 기부제", "알아두면 도움되는 꿀팁"),
                "page": 24,
            },
            {
                "title": "알아두면 좋은 꿀팁",
                "category": "benefit",
                "content": self._extract_section(content, "알아두면 도움되는 꿀팁", "Travel in Jecheon"),
                "page": 26,
            },
        ]

        # Create document chunks
        for idx, chunk_data in enumerate(chunks_data, start=1):
            if chunk_data["content"] and len(chunk_data["content"].strip()) > 50:
                doc_id = f"doc_{idx:03d}"
                self.chunks.append({
                    "doc_id": doc_id,
                    "title": chunk_data["title"],
                    "category": chunk_data["category"],
                    "content": chunk_data["content"].strip(),
                    "metadata": {
                        "page": chunk_data.get("page", 0),
                        "location": chunk_data.get("location", ""),
                        "address": chunk_data.get("address", ""),
                    },
                    "filename": f"{doc_id}_{chunk_data['title']}.txt",
                })

        return self.chunks

    def _extract_section(self, content: str, start_marker: str, end_marker: str) -> str:
        """Extract content between two markers."""
        try:
            start_idx = content.find(start_marker)
            if start_idx == -1:
                return ""

            end_idx = content.find(end_marker, start_idx + len(start_marker))
            if end_idx == -1:
                # Take rest of content if no end marker
                section = content[start_idx:]
            else:
                section = content[start_idx:end_idx]

            return section.strip()
        except Exception as e:
            print(f"Error extracting section {start_marker}: {e}")
            return ""

    def save_to_jsonl(self, output_path: str):
        """Save chunks to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        print(f"✅ Saved {len(self.chunks)} chunks to {output_path}")

    def print_summary(self):
        """Print summary of extracted chunks."""
        print(f"\n📊 Extraction Summary:")
        print(f"Total chunks: {len(self.chunks)}")

        # Category distribution
        category_counts = {}
        for chunk in self.chunks:
            cat = chunk["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print("\n📂 Category Distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {count}")

        print("\n📝 Sample Chunks:")
        for chunk in self.chunks[:3]:
            print(f"\n  [{chunk['doc_id']}] {chunk['title']} ({chunk['category']})")
            print(f"  Content preview: {chunk['content'][:100]}...")


def main():
    """Main execution function."""
    print("🚀 Starting Jecheon Tourism Document Chunking...")

    # Paths
    markdown_path = "/home/user/goodganglabs/data/processed/제천시관광정보책자.md"
    output_path = "/home/user/goodganglabs/data/chunks/documents.jsonl"

    # Initialize chunker
    chunker = JecheonDocumentChunker(markdown_path)

    # Extract chunks
    chunks = chunker.extract_chunks()

    # Save to JSONL
    chunker.save_to_jsonl(output_path)

    # Print summary
    chunker.print_summary()

    print("\n✅ Document chunking completed!")


if __name__ == "__main__":
    main()
