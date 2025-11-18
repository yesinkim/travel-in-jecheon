"""
Document Chunking Script for Jecheon Tourism Dataset (Optimized for RAG)

This script extracts meaningful chunks from the Jecheon tourism markdown file
with optimal size for RAG fine-tuning (300-2000 chars).

Key features:
- Minimum chunk size: 300 chars (sufficient context)
- Maximum chunk size: 2000 chars (avoid noise)
- Context-aware extraction (preserve semantic boundaries)

Output: data/chunks/documents.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


class JecheonDocumentChunker:
    """Extracts and chunks Jecheon tourism information optimized for RAG."""

    # RAG-optimized chunk sizes
    MIN_CHUNK_SIZE = 300   # chars (~75 tokens)
    MAX_CHUNK_SIZE = 2000  # chars (~500 tokens)

    CATEGORIES = {
        "transportation": ["시티투어", "관광택시", "관광주민증", "교통"],
        "tourism": ["의림지", "청풍", "배론", "박달재", "옥순봉", "케이블카", "관광지", "명소"],
        "food": ["맛집", "식당", "음식", "먹거리", "가스트로"],
        "accommodation": ["숙박", "리조트", "호텔", "게스트하우스"],
        "activity": ["트레킹", "체험", "축제", "레저", "걷기"],
        "culture": ["박물관", "성지", "문화", "역사", "유산"],
        "course": ["코스", "여행", "추천"],
        "benefit": ["인센티브", "할인", "혜택", "기부"],
    }

    def __init__(self, markdown_path: str):
        """Initialize chunker with markdown file path."""
        self.markdown_path = Path(markdown_path)
        self.chunks = []
        self.full_content = ""

    def read_markdown(self) -> str:
        """Read markdown file."""
        with open(self.markdown_path, 'r', encoding='utf-8') as f:
            return f.read()

    def clean_text(self, text: str) -> str:
        """Clean text by removing line numbers and extra whitespace."""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Remove line numbers (e.g., "    1→")
            cleaned_line = re.sub(r'^\s*\d+→', '', line)
            cleaned_lines.append(cleaned_line)

        # Join and clean up
        text = '\n'.join(cleaned_lines)
        # Remove multiple blank lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()

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

    def extract_between_patterns(self, content: str, start: str, end: str,
                                 include_start: bool = True) -> str:
        """Extract content between two patterns."""
        try:
            start_idx = content.find(start)
            if start_idx == -1:
                return ""

            if not include_start:
                start_idx += len(start)

            end_idx = content.find(end, start_idx + 1)
            if end_idx == -1:
                return content[start_idx:].strip()

            return content[start_idx:end_idx].strip()
        except Exception as e:
            return ""

    def ensure_min_size(self, content: str, title: str, context_before: str = "",
                       context_after: str = "") -> str:
        """
        Ensure chunk meets minimum size by adding context.
        If too small, add surrounding context.
        """
        if len(content) >= self.MIN_CHUNK_SIZE:
            return content

        # Add context to meet minimum size
        enhanced_content = content

        # Add context before if needed
        if context_before and len(enhanced_content) < self.MIN_CHUNK_SIZE:
            additional = context_before[-200:]  # Last 200 chars
            enhanced_content = f"{additional}\n\n{enhanced_content}"

        # Add context after if needed
        if context_after and len(enhanced_content) < self.MIN_CHUNK_SIZE:
            additional = context_after[:200]  # First 200 chars
            enhanced_content = f"{enhanced_content}\n\n{additional}"

        return enhanced_content

    def truncate_max_size(self, content: str) -> str:
        """Truncate content if it exceeds maximum size."""
        if len(content) <= self.MAX_CHUNK_SIZE:
            return content

        # Find last sentence boundary within limit
        truncated = content[:self.MAX_CHUNK_SIZE]
        last_period = max(truncated.rfind('.\n'), truncated.rfind('。\n'))

        if last_period > self.MIN_CHUNK_SIZE:
            return content[:last_period + 2].strip()

        return truncated.strip()

    def extract_tourist_site_enhanced(self, content: str, title: str,
                                      pattern: str) -> str:
        """Extract tourist site with enhanced context."""
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            return ""

        extracted = match.group(0).strip()

        # If too small, try to get more context
        if len(extracted) < self.MIN_CHUNK_SIZE:
            # Find position in full content
            pos = content.find(extracted)
            if pos != -1:
                # Get surrounding context
                start = max(0, pos - 200)
                end = min(len(content), pos + len(extracted) + 200)
                extended = content[start:end].strip()

                if len(extended) >= self.MIN_CHUNK_SIZE:
                    extracted = extended

        return self.truncate_max_size(extracted)

    def extract_chunks(self) -> List[Dict[str, Any]]:
        """Extract meaningful chunks from the markdown content."""
        content = self.read_markdown()
        content = self.clean_text(content)
        self.full_content = content

        chunks_data = []

        # 1. Transportation & Services
        chunks_data.extend([
            {
                "title": "디지털관광주민증",
                "category": "transportation",
                "content": self.extract_between_patterns(content, "디지털 관광주민증", "제천 시티투어"),
                "page": 4,
            },
            {
                "title": "제천 시티투어",
                "category": "transportation",
                "content": self.extract_between_patterns(content, "제천 시티투어", "제천 관광택시"),
                "page": 5,
            },
            {
                "title": "제천 관광택시",
                "category": "transportation",
                "content": self.extract_between_patterns(content, "제천 관광택시\n제천 토박이", "단체관광객 유치 인센티브"),
                "page": 6,
            },
            {
                "title": "단체관광객 인센티브",
                "category": "benefit",
                "content": self.extract_between_patterns(content, "단체관광객 유치 인센티브", "가스트로 투어"),
                "page": 7,
            },
            {
                "title": "가스트로 투어",
                "category": "food",
                "content": self.extract_between_patterns(content, "가스트로 투어", "모바일 바로가기"),
                "page": 8,
            },
        ])

        # 2. Tourist Sites (with enhanced context)
        tourist_sites = [
            {
                "title": "의림지·의림지역사박물관",
                "category": "tourism",
                "pattern": r"의림지(?:는|·의림지역사박물관).*?(?:제천시\s*(?:송학면|솔매로)[^\n]+)(?:\n[^\n]+){0,5}",
                "page": 12,
                "location": "송학면",
                "address": "제천시 송학면 의림대로 47길 7",
            },
            {
                "title": "배론성지",
                "category": "culture",
                "pattern": r"배론성지.*?제천시\s*봉양읍[^\n]+(?:\n[^\n]+){0,5}",
                "page": 12,
                "location": "봉양읍",
                "address": "제천시 봉양읍 배론성지길 296",
            },
            {
                "title": "박달재",
                "category": "tourism",
                "pattern": r"박달재.*?제천시\s*백운면[^\n]+(?:\n[^\n]+){0,5}",
                "page": 12,
                "location": "백운면",
                "address": "제천시 백운면 박달로 231",
            },
            {
                "title": "청풍호반 케이블카",
                "category": "tourism",
                "pattern": r"비봉산의?\s*(?:관광|풍경).*?청풍호반?\s*케이블카.*?제천시\s*청풍면[^\n]+(?:\n[^\n]+){0,3}",
                "page": 14,
                "location": "청풍면",
                "address": "제천시 청풍면 문화재길 166",
            },
            {
                "title": "청풍문화유산단지",
                "category": "culture",
                "pattern": r"청풍문화유산단지.*?제천시\s*청풍호로[^\n]+(?:\n[^\n]+){0,3}",
                "page": 14,
                "location": "청풍면",
                "address": "제천시 청풍호로 2048",
            },
            {
                "title": "청풍랜드",
                "category": "activity",
                "pattern": r"청풍랜드.*?제천시\s*청풍면[^\n]+(?:\n[^\n]+){0,3}",
                "page": 14,
                "location": "청풍면",
                "address": "제천시 청풍면 청풍호로50길 6",
            },
            {
                "title": "옥순봉 출렁다리",
                "category": "tourism",
                "pattern": r"(?:명승.*?)?옥순봉\s*출렁다리.*?제천시\s*수산면[^\n]+(?:\n[^\n]+){0,3}",
                "page": 14,
                "location": "수산면",
                "address": "제천시 수산면 옥순봉로342",
            },
            {
                "title": "국립 제천 치유의 숲",
                "category": "activity",
                "pattern": r"국립\s*제천\s*치유의\s*숲.*?제천시\s*청풍면[^\n]+(?:\n[^\n]+){0,3}",
                "page": 14,
                "location": "청풍면",
                "address": "제천시 청풍면 학현소야로 590",
            },
            {
                "title": "청풍호 자드락길",
                "category": "activity",
                "pattern": r"청풍호\s*자드락길.*?제천시\s*수산면[^\n]+(?:\n[^\n]+){0,3}",
                "page": 14,
                "location": "수산면",
                "address": "제천시 수산면 옥순봉로 6길 3",
            },
        ]

        for site in tourist_sites:
            extracted = self.extract_tourist_site_enhanced(
                content, site["title"], site["pattern"]
            )
            if extracted:
                chunks_data.append({
                    "title": site["title"],
                    "category": site["category"],
                    "content": extracted,
                    "page": site["page"],
                    "location": site.get("location", ""),
                    "address": site.get("address", ""),
                })

        # 3. Additional sites
        additional_sites = [
            ("삼한의 초록길", "activity", r"삼한의\s*초록길.*?(?:제천시\s*성봉로|건강과\s*심리치유)[^\n]*(?:\n[^\n]+){0,5}"),
            ("제천의 축제", "activity", r"제천의\s*축제.*?(?:청풍호\s*벚꽃축제|봉양박달콩축제).*?(?:\n[^\n]+){0,8}"),
        ]

        for title, category, pattern in additional_sites:
            extracted = self.extract_tourist_site_enhanced(content, title, pattern)
            if extracted:
                chunks_data.append({
                    "title": title,
                    "category": category,
                    "content": extracted,
                    "page": 12,
                })

        # 4. Trekking & Activities
        trekking_content = self.extract_between_patterns(content, "트래킹·걷기 좋은곳", "코스여행 추천")
        if trekking_content:
            chunks_data.append({
                "title": "트레킹·걷기 좋은 곳",
                "category": "activity",
                "content": trekking_content,
                "page": 16,
            })

        # 5. Travel Courses
        chunks_data.extend([
            {
                "title": "제천 1일 코스",
                "category": "course",
                "content": self.extract_between_patterns(content, "1일 코스", "1박 2일 코스"),
                "page": 17,
            },
            {
                "title": "제천 1박 2일 코스",
                "category": "course",
                "content": self.extract_between_patterns(content, "1박 2일 코스", "휴양·힐링 코스"),
                "page": 17,
            },
            {
                "title": "슬로시티 힐링 코스",
                "category": "course",
                "content": self.extract_between_patterns(content, "#슬로시티 코스", "#북부·서부권 코스"),
                "page": 17,
            },
            {
                "title": "백운권 힐링 코스",
                "category": "course",
                "content": self.extract_between_patterns(content, "#북부·서부권 코스", "문화·역사 코스"),
                "page": 17,
            },
            {
                "title": "불교 순례 코스",
                "category": "course",
                "content": self.extract_between_patterns(content, "#불교 코스", "#천주교 코스"),
                "page": 18,
            },
            {
                "title": "천주교 순례 코스",
                "category": "course",
                "content": self.extract_between_patterns(content, "#천주교 코스", "#기독교 코스"),
                "page": 18,
            },
        ])

        # 6. Food & Restaurants
        chunks_data.extend([
            {
                "title": "제천 맛집 안내",
                "category": "food",
                "content": self.extract_between_patterns(content, "제천맛집", "북부권 ("),
                "page": 20,
            },
            {
                "title": "북부권 맛집",
                "category": "food",
                "content": self.extract_between_patterns(content, "북부권 (9)", "남부권 ("),
                "page": 21,
            },
            {
                "title": "청풍권 맛집",
                "category": "food",
                "content": self.extract_between_patterns(content, "청풍권 (11)", "주요 숙박시설"),
                "page": 21,
            },
        ])

        # 7. Accommodation
        accommodations = [
            ("포레스트 리솜", "제천시 백운면 금봉로 365", "043, 649, 6000"),
            ("청풍리조트", "제천시 청풍면 청풍호로 1798", "043, 640, 7000"),
            ("ES리조트", "제천시 수산면 옥순봉로 1248", "043, 648, 0480"),
            ("서울관광호텔", "제천시 의림대로13길 10", "043, 651, 8000"),
        ]

        for name, address, phone in accommodations:
            # Find accommodation with context
            pattern = f"{name}.*?{phone}"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(0).strip()
                # Add description if too short
                if len(extracted) < self.MIN_CHUNK_SIZE:
                    # Add accommodation type description
                    desc = f"{name}은(는) 제천의 주요 숙박시설입니다.\n주소: {address}\n연락처: {phone}"
                    extracted = f"{desc}\n\n{extracted}"

                chunks_data.append({
                    "title": name,
                    "category": "accommodation",
                    "content": self.truncate_max_size(extracted),
                    "page": 22,
                    "address": address,
                })

        # 8. Benefits & Tips
        chunks_data.extend([
            {
                "title": "고향사랑 기부제",
                "category": "benefit",
                "content": self.extract_between_patterns(content, "고향사랑 기부제", "알아두면 도움되는 꿀팁"),
                "page": 24,
            },
            {
                "title": "알아두면 좋은 정보",
                "category": "benefit",
                "content": self.extract_between_patterns(content, "알아두면 도움되는 꿀팁", "Travel in Jecheon"),
                "page": 26,
            },
        ])

        # Create document chunks with size validation
        doc_id = 1
        for chunk_data in chunks_data:
            content_text = chunk_data.get("content", "")
            if not content_text or len(content_text.strip()) < 50:
                continue

            # Ensure minimum size
            title = chunk_data["title"]
            if len(content_text) < self.MIN_CHUNK_SIZE:
                # Add title as context if needed
                enhanced = f"# {title}\n\n{content_text}"
                if len(enhanced) < self.MIN_CHUNK_SIZE and chunk_data.get("address"):
                    enhanced += f"\n\n위치: {chunk_data['address']}"
                content_text = enhanced

            # Truncate if too large
            content_text = self.truncate_max_size(content_text)

            self.chunks.append({
                "doc_id": f"doc_{doc_id:03d}",
                "title": chunk_data["title"],
                "category": chunk_data["category"],
                "content": content_text.strip(),
                "metadata": {
                    "page": chunk_data.get("page", 0),
                    "location": chunk_data.get("location", ""),
                    "address": chunk_data.get("address", ""),
                },
                "filename": f"doc_{doc_id:03d}_{chunk_data['title']}.txt",
            })
            doc_id += 1

        return self.chunks

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

        # Size statistics
        sizes = [len(chunk["content"]) for chunk in self.chunks]
        if sizes:
            import statistics
            print(f"\n📏 Chunk Size Statistics:")
            print(f"  Average: {statistics.mean(sizes):.0f} chars")
            print(f"  Median: {statistics.median(sizes):.0f} chars")
            print(f"  Min: {min(sizes)} chars")
            print(f"  Max: {max(sizes)} chars")

            # Size distribution
            print(f"\n📊 Size Distribution:")
            print(f"  < 300 chars: {sum(1 for s in sizes if s < 300)} ⚠️")
            print(f"  300-1000 chars: {sum(1 for s in sizes if 300 <= s < 1000)} ✅")
            print(f"  1000-2000 chars: {sum(1 for s in sizes if 1000 <= s < 2000)} ✅")
            print(f"  > 2000 chars: {sum(1 for s in sizes if s >= 2000)} ⚠️")

        # Category distribution
        category_counts = {}
        for chunk in self.chunks:
            cat = chunk["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print("\n📂 Category Distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {count}")

        print("\n📝 Sample Chunks:")
        for chunk in self.chunks[:5]:
            addr = chunk['metadata'].get('address', '')
            addr_str = f" | {addr}" if addr else ""
            size = len(chunk['content'])
            print(f"  [{chunk['doc_id']}] {chunk['title']} ({size} chars){addr_str}")


def main():
    """Main execution function."""
    print("🚀 Starting Jecheon Tourism Document Chunking (RAG-Optimized)...")

    # Paths (using OCR-corrected markdown)
    markdown_path = "/home/user/goodganglabs/data/processed/제천시관광정보책자_corrected.md"
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
