"""
Document Chunking Script for Jecheon Tourism Dataset (Enhanced Version)

This script extracts meaningful chunks from the Jecheon tourism markdown file
with more granular segmentation. Overlapping is allowed for better coverage.

Output: data/chunks/documents.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


class JecheonDocumentChunker:
    """Extracts and chunks Jecheon tourism information with fine granularity."""

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

    def extract_tourist_sites(self, content: str) -> List[Dict[str, Any]]:
        """Extract individual tourist sites as separate chunks."""
        sites = []

        # Pattern: Site name followed by address
        site_patterns = [
            # Major sites with full details
            (r"의림지·의림지역사박물관.*?(?=배론성지|$)", "의림지·의림지역사박물관", "tourism", 12, "송학면", "제천시 송학면 의림대로 47길 7"),
            (r"배론성지.*?(?=박달재|$)", "배론성지", "culture", 12, "봉양읍", "제천시 봉양읍 배론성지길 296"),
            (r"박달재(?:는|\.)[^제]*?제천시[^\n]+(?:\n[^\n제]+){0,3}", "박달재", "tourism", 12, "백운면", "제천시 백운면 박달로 231"),
            (r"제천한방엑스포\s*공원.*?(?=의림지|$)", "제천한방엑스포 공원", "culture", 12, "", "제천시 한방엑스포로 19"),
            (r"청풍호반\s*케이블카.*?제천시\s*청풍면[^\n]+", "청풍호반 케이블카", "tourism", 14, "청풍면", "제천시 청풍면 문화재길 166"),
            (r"청풍문화유산단지.*?제천시\s*청풍호로[^\n]+", "청풍문화유산단지", "culture", 14, "청풍면", "제천시 청풍호로 2048"),
            (r"청풍랜드.*?제천시\s*청풍면[^\n]+", "청풍랜드", "activity", 14, "청풍면", "제천시 청풍면 청풍호로50길 6"),
            (r"옥순봉\s*출렁다리.*?제천시\s*수산면[^\n]+", "옥순봉 출렁다리", "tourism", 14, "수산면", "제천시 수산면 옥순봉로342"),
            (r"국립\s*제천\s*치유의\s*숲.*?제천시\s*청풍면[^\n]+", "국립 제천 치유의 숲", "activity", 14, "청풍면", "제천시 청풍면 학현소야로 590"),
            (r"청풍호\s*자드락길.*?제천시\s*수산면[^\n]+", "청풍호 자드락길", "activity", 14, "수산면", "제천시 수산면 옥순봉로 6길 3"),
        ]

        for pattern, title, category, page, location, address in site_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                sites.append({
                    "title": title,
                    "category": category,
                    "content": match.group(0).strip(),
                    "page": page,
                    "location": location,
                    "address": address,
                })

        # Additional sites from secondary sections
        additional_sites = [
            ("의림지 수리공원", "activity", r"의림지\s*수리공원.*?(?:운영기간|제천시\s*모산동)[^\n]*(?:\n[^\n]+){0,5}"),
            ("삼한의 초록길", "activity", r"삼한의\s*초록길.*?(?:제천시\s*성봉로|km\s*길이)[^\n]*(?:\n[^\n]+){0,3}"),
            ("교동민화마을", "culture", r"교동민화마을.*?제천시\s*용두로[^\n]+(?:\n[^\n]+){0,2}"),
            ("모산비행장", "tourism", r"모산비행장.*?제천시\s*고암동[^\n]+(?:\n[^\n]+){0,2}"),
            ("아열대 스마트온실", "activity", r"아열대\s*스마트온실.*?제천시\s*봉양읍[^\n]+(?:\n[^\n]+){0,3}"),
            ("한국차문화박물관", "culture", r"한국차문화박물관.*?제천시\s*금학로[^\n]+(?:\n[^\n]+){0,2}"),
            ("벌새꽃돌과학관", "culture", r"벌새꽃돌과학관.*?제천시\s*봉양읍[^\n]+(?:\n[^\n]+){0,3}"),
        ]

        for title, category, pattern in additional_sites:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                # Extract address from content
                address_match = re.search(r'제천시[^\n]+', match.group(0))
                address = address_match.group(0) if address_match else ""

                sites.append({
                    "title": title,
                    "category": category,
                    "content": match.group(0).strip(),
                    "page": 12,  # Default page
                    "location": "",
                    "address": address,
                })

        return sites

    def extract_chunks(self) -> List[Dict[str, Any]]:
        """Extract meaningful chunks from the markdown content."""
        content = self.read_markdown()
        content = self.clean_text(content)

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

        # 2. Extract tourist sites (granular)
        tourist_sites = self.extract_tourist_sites(content)
        for site in tourist_sites:
            chunks_data.append(site)

        # 3. Trekking & Activities
        chunks_data.append({
            "title": "트레킹·걷기 좋은 곳",
            "category": "activity",
            "content": self.extract_between_patterns(content, "트래킹·걷기 좋은곳", "코스여행 추천"),
            "page": 16,
        })

        # 4. Travel Courses (each course separately)
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
                "title": "문화·역사 코스",
                "category": "course",
                "content": self.extract_between_patterns(content, "문화·역사 코스", "종교여행 코스"),
                "page": 18,
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
            {
                "title": "기독교 순례 코스",
                "category": "course",
                "content": self.extract_between_patterns(content, "#기독교 코스", "#유교·의병문화 코스"),
                "page": 18,
            },
        ])

        # 5. Food & Restaurants
        chunks_data.extend([
            {
                "title": "제천 맛집 브랜드 (약채락·의림지에코닉)",
                "category": "food",
                "content": self.extract_between_patterns(content, "제천맛집", "시내권 ("),
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

        # 6. Accommodation (individual facilities)
        accommodations = [
            ("포레스트 리솜", "제천시 백운면 금봉로 365", "043, 649, 6000"),
            ("청풍리조트", "제천시 청풍면 청풍호로 1798", "043, 640, 7000"),
            ("ES리조트", "제천시 수산면 옥순봉로 1248", "043, 648, 0480"),
            ("서울관광호텔", "제천시 의림대로13길 10", "043, 651, 8000"),
        ]

        for name, address, phone in accommodations:
            pattern = f"{name}.*?{address}.*?{phone}"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                chunks_data.append({
                    "title": name,
                    "category": "accommodation",
                    "content": match.group(0).strip(),
                    "page": 22,
                    "address": address,
                })

        # 7. Benefits & Tips
        chunks_data.extend([
            {
                "title": "고향사랑 기부제",
                "category": "benefit",
                "content": self.extract_between_patterns(content, "고향사랑 기부제", "알아두면 도움되는 꿀팁"),
                "page": 24,
            },
            {
                "title": "청풍호 수경분수 운영시간",
                "category": "benefit",
                "content": self.extract_between_patterns(content, "청풍호조경분수", "의림지미디어파사드"),
                "page": 26,
            },
            {
                "title": "의림지 미디어파사드 운영시간",
                "category": "benefit",
                "content": self.extract_between_patterns(content, "의림지미디어파사드", "육삼륙 관광단지"),
                "page": 26,
            },
        ])

        # 8. Festivals
        festivals_content = self.extract_between_patterns(content, "제천의 축제", "미리보는 여행지")
        if festivals_content:
            chunks_data.append({
                "title": "제천의 축제",
                "category": "activity",
                "content": festivals_content,
                "page": 11,
            })

        # Create document chunks with proper doc_ids
        doc_id = 1
        for chunk_data in chunks_data:
            if chunk_data.get("content") and len(chunk_data["content"].strip()) > 30:
                self.chunks.append({
                    "doc_id": f"doc_{doc_id:03d}",
                    "title": chunk_data["title"],
                    "category": chunk_data["category"],
                    "content": chunk_data["content"].strip(),
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

        # Category distribution
        category_counts = {}
        for chunk in self.chunks:
            cat = chunk["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print("\n📂 Category Distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {count}")

        print("\n📝 All Chunks:")
        for chunk in self.chunks:
            addr = chunk['metadata'].get('address', '')
            addr_str = f" | {addr}" if addr else ""
            print(f"  [{chunk['doc_id']}] {chunk['title']} ({chunk['category']}){addr_str}")


def main():
    """Main execution function."""
    print("🚀 Starting Jecheon Tourism Document Chunking (Enhanced)...")

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
