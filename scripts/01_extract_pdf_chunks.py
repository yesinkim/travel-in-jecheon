"""
Document Chunking Script for Jecheon Tourism Dataset (TOC-Guided Version)

개선사항:
1. TOC (Table of Contents) parsing and validation
2. Section-based automatic chunking (### 헤더 기반)
3. PART 기반 그룹화 (PART 1-5)
4. No hardcoded tourist sites - fully automatic
5. Dynamic category assignment based on PART and keywords
6. Better metadata extraction

Input: data/processed/제천시관광정보책자.md
Output: data/chunks/documents.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics
from dataclasses import dataclass


@dataclass
class TOCEntry:
    """Table of Contents entry."""
    title: str
    page: int
    part: str  # PART 1, PART 2, etc.
    part_number: int  # 1, 2, 3, 4, 5


class TOCGuidedChunker:
    """TOC-guided document chunker for reliable section extraction."""

    # RAG-optimized chunk sizes
    MIN_CHUNK_SIZE = 300   # chars (~75 tokens)
    MAX_CHUNK_SIZE = 2000  # chars (~500 tokens)

    # PART-based category mapping
    PART_CATEGORIES = {
        1: "transportation",  # 출발 전 준비
        2: "tourism",         # 미리보는 여행지
        3: "food",            # 제천에서의 맛있는 하루
        4: "accommodation",   # 편안한 휴식과 숙소
        5: "benefit",         # 함께하는 제천
    }

    # Keyword-based category refinement
    CATEGORY_KEYWORDS = {
        "transportation": ["시티투어", "관광택시", "교통", "버스", "택시"],
        "tourism": ["관광지", "명소", "10경", "케이블카", "출렁다리", "박물관", "성지"],
        "food": ["맛집", "식당", "음식", "먹거리", "가스트로"],
        "accommodation": ["숙박", "리조트", "호텔"],
        "activity": ["트레킹", "체험", "축제", "걷기", "코스", "힐링"],
        "benefit": ["인센티브", "할인", "혜택", "기부", "주민증", "QR", "꿀팁"],
    }

    def __init__(self, markdown_path: str):
        """Initialize chunker with markdown file path."""
        self.markdown_path = Path(markdown_path)
        self.chunks = []
        self.full_content = ""
        self.toc_entries: List[TOCEntry] = []

    def read_markdown(self) -> str:
        """Read markdown file."""
        with open(self.markdown_path, 'r', encoding='utf-8') as f:
            return f.read()

    def parse_toc(self, content: str) -> List[TOCEntry]:
        """
        Parse table of contents from markdown.

        Expected format:
        **PART 1**
        -   디지털관광주민증 04
        -   제천 시티투어 05
        """
        toc_entries = []

        # Find TOC section
        toc_match = re.search(r'## contents\s*\n(.*?)\n---', content, re.DOTALL | re.IGNORECASE)
        if not toc_match:
            print("⚠️  Warning: TOC not found, using fallback method")
            return []

        toc_text = toc_match.group(1)

        current_part = None
        current_part_number = None

        for line in toc_text.split('\n'):
            # Match PART header: **PART 1**
            part_match = re.match(r'\*\*PART\s+(\d+)\*\*', line)
            if part_match:
                current_part_number = int(part_match.group(1))
                current_part = f"PART {current_part_number}"
                continue

            # Match TOC entry: -   디지털관광주민증 04
            entry_match = re.match(r'-\s+(.+?)\s+(\d+)', line)
            if entry_match and current_part:
                title = entry_match.group(1).strip('*')  # Remove ** markers
                page = int(entry_match.group(2))

                # Skip section headers (볼드 처리된 것들)
                if title.startswith('**'):
                    continue

                toc_entries.append(TOCEntry(
                    title=title,
                    page=page,
                    part=current_part,
                    part_number=current_part_number
                ))

        self.toc_entries = toc_entries
        return toc_entries

    def get_category_from_part_and_keywords(self, part_number: int, title: str, content: str) -> str:
        """Determine category from PART number and keyword matching."""
        # Start with PART-based category
        base_category = self.PART_CATEGORIES.get(part_number, "general")

        # Refine with keywords
        text = (title + " " + content).lower()

        category_scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            if score > 0:
                category_scores[category] = score

        # If keyword match found, use it; otherwise use PART-based category
        if category_scores:
            keyword_category = max(category_scores.items(), key=lambda x: x[1])[0]
            # Override base category if keyword match is strong (score >= 2)
            if category_scores[keyword_category] >= 2:
                return keyword_category

        return base_category

    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from content (address, phone, URL, etc.)."""
        metadata = {}

        # Extract address (제천시로 시작하는 주소)
        address_match = re.search(r'제천시\s+\S+\s+[^\n]{5,50}', content)
        if address_match:
            addr = address_match.group(0).strip()
            # Clean up common artifacts
            addr = re.sub(r'\s+', ' ', addr)
            metadata['address'] = addr

        # Extract phone number
        phone_match = re.search(r'043[.\s,]*\d{3}[.\s,]*\d{4}', content)
        if phone_match:
            phone = phone_match.group(0)
            # Normalize format
            phone = re.sub(r'[.\s,]+', '-', phone)
            metadata['phone'] = phone

        # Extract URL
        url_match = re.search(r'\[([a-z0-9\-\.]+\.(?:go\.kr|com|net|org))\]|(?:https?://)?([a-z0-9\-\.]+\.(?:go\.kr|com))', content, re.IGNORECASE)
        if url_match:
            metadata['url'] = url_match.group(1) or url_match.group(2)

        # Extract price/cost information
        price_match = re.search(r'(\d{1,3}(?:,\d{3})*원)', content)
        if price_match:
            metadata['price'] = price_match.group(1)

        return metadata

    def find_section_content(self, title: str, content: str) -> Optional[str]:
        """
        Find section content by title using ### header.

        Returns content between this section and next section.
        """
        # Normalize title for better matching (remove spaces, special chars)
        def normalize(text):
            # Remove spaces, &, ·, 코드, 모음 etc.
            text = re.sub(r'[\s&·]+', '', text.lower())
            # Remove common words that may differ between TOC and actual title
            text = re.sub(r'(코드|모음|유치|도움되는|좋은)', '', text)
            return text

        normalized_title = normalize(title)

        # Find all ### sections
        sections = re.findall(r'###\s+(.+?)\n(.*?)(?=\n###|$)', content, re.DOTALL)

        # Try to find best match
        best_match = None
        best_score = 0

        for section_title, section_content in sections:
            normalized_section = normalize(section_title)

            # Exact match (after normalization)
            if normalized_title == normalized_section:
                return section_content.strip()

            # Partial match - check if one contains the other (both directions)
            if normalized_title in normalized_section:
                # TOC title is substring of section title
                score = len(normalized_title)
                if score > best_score:
                    best_score = score
                    best_match = section_content.strip()
            elif normalized_section in normalized_title:
                # Section title is substring of TOC title
                score = len(normalized_section)
                if score > best_score:
                    best_score = score
                    best_match = section_content.strip()

        # Accept match if at least 60% of shorter title matches
        min_title_len = min(len(normalized_title), 10)
        if best_match and best_score >= min_title_len * 0.6:
            return best_match

        return None

    def split_large_content(self, title: str, content: str, max_size: int) -> List[str]:
        """
        Split large content into smaller chunks while preserving context.

        Strategy:
        1. Split by paragraphs (double newline)
        2. Each chunk includes title for context
        3. Try to keep chunks between MIN_CHUNK_SIZE and MAX_CHUNK_SIZE
        """
        if len(content) <= max_size:
            return [content]

        chunks = []

        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', content)

        current_chunk = f"# {title}\n\n"

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds max, save current chunk
            if len(current_chunk) + len(para) + 2 > max_size:
                if len(current_chunk) > self.MIN_CHUNK_SIZE:
                    chunks.append(current_chunk.strip())
                    current_chunk = f"# {title}\n\n{para}\n\n"
                else:
                    # Current chunk too small, add anyway
                    current_chunk += para + "\n\n"
            else:
                current_chunk += para + "\n\n"

        # Add remaining content
        if len(current_chunk.strip()) > len(f"# {title}"):
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content]

    def ensure_min_size(self, content: str, title: str, metadata: Dict[str, str]) -> str:
        """Ensure chunk meets minimum size by adding context."""
        if len(content) >= self.MIN_CHUNK_SIZE:
            return content

        # Add title as header
        enhanced = f"# {title}\n\n{content}"

        # Still too small? Add metadata as context
        if len(enhanced) < self.MIN_CHUNK_SIZE and metadata:
            meta_text = "\n\n"
            if metadata.get('address'):
                meta_text += f"주소: {metadata['address']}\n"
            if metadata.get('phone'):
                meta_text += f"연락처: {metadata['phone']}\n"
            if metadata.get('url'):
                meta_text += f"홈페이지: {metadata['url']}\n"

            if meta_text.strip():
                enhanced += meta_text

        return enhanced

    def create_chunks_from_toc(self) -> List[Dict[str, Any]]:
        """
        Create document chunks guided by TOC.

        Process:
        1. Parse TOC to get all section titles
        2. For each TOC entry, find corresponding ### section
        3. Extract content
        4. Add metadata and categorize
        5. Split if too large
        """
        content = self.read_markdown()
        self.full_content = content

        print("📋 Parsing Table of Contents...")
        toc_entries = self.parse_toc(content)

        if not toc_entries:
            print("⚠️  TOC parsing failed, falling back to all ### sections")
            return self.create_chunks_fallback()

        print(f"✅ Found {len(toc_entries)} entries in TOC")

        chunks_data = []
        doc_id = 1

        for toc_entry in toc_entries:
            title = toc_entry.title

            print(f"\n🔍 Processing: {title} ({toc_entry.part})")

            # Find section content
            section_content = self.find_section_content(title, content)

            if not section_content:
                print(f"  ⚠️  Content not found for: {title}")
                continue

            if len(section_content.strip()) < 50:
                print(f"  ⏭️  Skipping (too small): {title}")
                continue

            # Extract metadata
            metadata = self.extract_metadata(section_content)
            metadata['page'] = toc_entry.page
            metadata['part'] = toc_entry.part

            # Determine category
            category = self.get_category_from_part_and_keywords(
                toc_entry.part_number,
                title,
                section_content
            )

            # Handle large sections
            if len(section_content) > self.MAX_CHUNK_SIZE:
                print(f"  ✂️  Splitting large section ({len(section_content)} chars)")
                chunk_contents = self.split_large_content(title, section_content, self.MAX_CHUNK_SIZE)
            else:
                # Ensure minimum size
                chunk_contents = [self.ensure_min_size(section_content, title, metadata)]

            # Create chunks
            for i, chunk_content in enumerate(chunk_contents):
                chunk_title = title if len(chunk_contents) == 1 else f"{title} (Part {i+1})"

                chunk = {
                    "doc_id": f"doc_{doc_id:03d}",
                    "title": chunk_title,
                    "category": category,
                    "content": chunk_content.strip(),
                    "metadata": metadata,
                    "filename": f"doc_{doc_id:03d}_{title.replace('/', '_').replace('·', '_')[:50]}.txt",
                    "size": len(chunk_content),
                }

                chunks_data.append(chunk)
                print(f"  ✅ [{chunk['doc_id']}] {chunk_title[:45]} ({category}, {chunk['size']} chars)")
                doc_id += 1

        self.chunks = chunks_data
        return chunks_data

    def create_chunks_fallback(self) -> List[Dict[str, Any]]:
        """
        Fallback method: extract all ### sections if TOC parsing fails.
        """
        content = self.full_content if self.full_content else self.read_markdown()

        chunks_data = []
        doc_id = 1

        # Find all ### sections
        sections = re.findall(r'###\s+(.+?)\n(.*?)(?=\n###|$)', content, re.DOTALL)

        for title, section_content in sections:
            title = title.strip()
            section_content = section_content.strip()

            if len(section_content) < 50:
                continue

            # Extract metadata
            metadata = self.extract_metadata(section_content)

            # Determine category (use keywords only, no PART info)
            category = "general"
            text = (title + " " + section_content).lower()
            category_scores = {}

            for cat, keywords in self.CATEGORY_KEYWORDS.items():
                score = sum(1 for kw in keywords if kw in text)
                if score > 0:
                    category_scores[cat] = score

            if category_scores:
                category = max(category_scores.items(), key=lambda x: x[1])[0]

            chunk = {
                "doc_id": f"doc_{doc_id:03d}",
                "title": title,
                "category": category,
                "content": section_content,
                "metadata": metadata,
                "filename": f"doc_{doc_id:03d}_{title[:50]}.txt",
                "size": len(section_content),
            }

            chunks_data.append(chunk)
            doc_id += 1

        self.chunks = chunks_data
        return chunks_data

    def save_to_jsonl(self, output_path: str):
        """Save chunks to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        print(f"\n✅ Saved {len(self.chunks)} chunks to {output_path}")

    def print_summary(self):
        """Print comprehensive summary of extracted chunks."""
        print(f"\n{'='*70}")
        print("📊 CHUNKING SUMMARY")
        print(f"{'='*70}")
        print(f"Total chunks: {len(self.chunks)}")
        print(f"TOC entries found: {len(self.toc_entries)}")

        # Size statistics
        sizes = [chunk["size"] for chunk in self.chunks]
        if sizes:
            print(f"\n📏 Chunk Size Statistics:")
            print(f"  Average: {statistics.mean(sizes):.0f} chars")
            print(f"  Median: {statistics.median(sizes):.0f} chars")
            print(f"  Min: {min(sizes)} chars")
            print(f"  Max: {max(sizes)} chars")

            # Size distribution
            print(f"\n📊 Size Distribution:")
            under_300 = sum(1 for s in sizes if s < 300)
            mid_range = sum(1 for s in sizes if 300 <= s < 1000)
            large_range = sum(1 for s in sizes if 1000 <= s < 2000)
            over_2000 = sum(1 for s in sizes if s >= 2000)

            print(f"  < 300 chars: {under_300} {'⚠️' if under_300 > 0 else '✅'}")
            print(f"  300-1000 chars: {mid_range} ✅")
            print(f"  1000-2000 chars: {large_range} ✅")
            print(f"  > 2000 chars: {over_2000} {'⚠️' if over_2000 > 0 else '✅'}")

        # Category distribution
        category_counts = {}
        for chunk in self.chunks:
            cat = chunk["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print("\n📂 Category Distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {count}")

        # PART distribution
        part_counts = {}
        for chunk in self.chunks:
            part = chunk['metadata'].get('part', 'Unknown')
            part_counts[part] = part_counts.get(part, 0) + 1

        print("\n📚 PART Distribution:")
        for part in ['PART 1', 'PART 2', 'PART 3', 'PART 4', 'PART 5']:
            count = part_counts.get(part, 0)
            print(f"  - {part}: {count}")

        print("\n📝 Sample Chunks (first 5):")
        for chunk in self.chunks[:5]:
            addr = chunk['metadata'].get('address', '')
            addr_str = f" | {addr[:30]}..." if addr else ""
            part = chunk['metadata'].get('part', 'N/A')
            print(f"  [{chunk['doc_id']}] {chunk['title'][:40]} ({chunk['category']}, {part}){addr_str}")

        # Validation warnings
        print(f"\n{'='*70}")
        print("⚠️  VALIDATION WARNINGS")
        print(f"{'='*70}")

        warnings = []
        if under_300 > 0:
            warnings.append(f"- {under_300} chunks below minimum size (300 chars)")
        if over_2000 > 0:
            warnings.append(f"- {over_2000} chunks exceed maximum size (2000 chars)")
        if len(self.chunks) < len(self.toc_entries):
            warnings.append(f"- Missing chunks: TOC has {len(self.toc_entries)} entries but only {len(self.chunks)} chunks created")

        if warnings:
            for warning in warnings:
                print(warning)
        else:
            print("✅ No warnings - all chunks are within optimal size range!")


def main():
    """Main execution function."""
    print("\n🚀 Starting TOC-Guided Jecheon Tourism Document Chunking")
    print("="*70)

    # Paths
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    markdown_path = os.path.join(project_root, "data", "processed", "제천시관광정보책자.md")
    output_path = os.path.join(project_root, "data", "chunks", "documents.jsonl")

    print(f"📄 Input: {markdown_path}")
    print(f"💾 Output: {output_path}")

    # Initialize chunker
    chunker = TOCGuidedChunker(markdown_path)

    # Create chunks (TOC-guided)
    chunks = chunker.create_chunks_from_toc()

    # Save to JSONL
    chunker.save_to_jsonl(output_path)

    # Print summary
    chunker.print_summary()

    print("\n" + "="*70)
    print("✅ Document chunking completed!")
    print("="*70)


if __name__ == "__main__":
    main()
