"""
Create Clean Markdown from PyMuPDF TXT

Uses PyMuPDF as base (more accurate) and creates a clean markdown file
suitable for RAG dataset generation.

Input: data/processed/full_extraction/pymupdf_full.txt
Output: data/processed/제천시관광정보책자_clean.md
"""

import re
from pathlib import Path
from typing import List, Dict


class CleanMarkdownCreator:
    """Creates clean markdown from PyMuPDF extraction."""

    def __init__(self, txt_path: str):
        """Initialize with PyMuPDF txt file path."""
        self.txt_path = Path(txt_path)
        self.content = ""
        self.pages = {}

    def read_txt(self):
        """Read PyMuPDF txt file."""
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
        print(f"✅ Read TXT file: {len(self.content)} chars")

    def extract_pages(self) -> Dict[int, str]:
        """Extract content by page number."""
        pages = {}

        # Split by page markers
        page_pattern = r'={60}\nPage (\d+)\n={60}\n(.*?)(?=\n={60}\nPage \d+|$)'
        matches = re.findall(page_pattern, self.content, re.DOTALL)

        for page_num_str, page_content in matches:
            page_num = int(page_num_str)
            pages[page_num] = page_content.strip()

        print(f"✅ Extracted {len(pages)} pages")
        return pages

    def clean_page_content(self, content: str) -> str:
        """Clean individual page content."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

        # Fix common OCR issues
        content = content.replace('|상품소개|', '\n**상품소개**\n')
        content = content.replace('|요금안내|', '\n**요금안내**\n')
        content = content.replace('|예약안내|', '\n**예약안내**\n')
        content = content.replace('|발급방법|', '\n**발급방법**\n')
        content = content.replace('|주요혜택|', '\n**주요혜택**\n')
        content = content.replace('|기본코스|', '\n**기본코스**\n')
        content = content.replace('|지원금액|', '\n**지원금액**\n')
        content = content.replace('|코스안내|', '\n**코스안내**\n')

        return content.strip()

    def structure_content(self, pages: Dict[int, str]) -> str:
        """Structure pages into logical sections."""
        markdown = []

        # Title (Page 1)
        if 1 in pages:
            markdown.append("# 2025 제천 여행 가이드\n")
            markdown.append("Travel in Jecheon\n")
            markdown.append("\n---\n\n")

        # Table of Contents (Page 2)
        if 2 in pages:
            markdown.append("## 목차 (Contents)\n\n")
            markdown.append(self.clean_page_content(pages[2]))
            markdown.append("\n\n---\n\n")

        # Part 1: 출발 전 준비 (Pages 3-9)
        markdown.append("# PART 1: 출발 전 준비\n\n")

        for page_num in range(3, 10):
            if page_num in pages:
                content = self.clean_page_content(pages[page_num])
                if content:
                    # Detect section titles
                    if "디지털" in content and "관광주민증" in content:
                        markdown.append("## 디지털관광주민증\n\n")
                    elif "시티투어" in content:
                        markdown.append("## 제천 시티투어\n\n")
                    elif "관광택시" in content and "시티투어" not in content:
                        markdown.append("## 제천 관광택시\n\n")
                    elif "인센티브" in content:
                        markdown.append("## 단체관광객 유치 인센티브\n\n")
                    elif "가스트로" in content:
                        markdown.append("## 가스트로 투어\n\n")

                    markdown.append(content)
                    markdown.append("\n\n")

        # Part 2: 미리보는 여행지 (Pages 10-19)
        markdown.append("---\n\n# PART 2: 미리보는 여행지\n\n")

        for page_num in range(10, 20):
            if page_num in pages:
                content = self.clean_page_content(pages[page_num])
                if content:
                    # Detect major sections
                    if "축제" in content and len(content) < 500:
                        markdown.append("## 제천의 축제\n\n")
                    elif "북부" in content and "주요 관광지" in content:
                        markdown.append("## 북부·의림·도심권역 주요 관광지\n\n")
                    elif "남부" in content and "주요 관광지" in content:
                        markdown.append("## 남부권역 주요 관광지\n\n")
                    elif "트레킹" in content or "걷기" in content:
                        markdown.append("## 트레킹·걷기 좋은 곳\n\n")
                    elif "코스 여행" in content or "추천" in content:
                        markdown.append("## 코스 여행 추천\n\n")

                    markdown.append(content)
                    markdown.append("\n\n")

        # Part 3: 맛집 (Pages 20-21)
        markdown.append("---\n\n# PART 3: 제천에서의 맛있는 하루\n\n")

        for page_num in range(20, 22):
            if page_num in pages:
                content = self.clean_page_content(pages[page_num])
                if content:
                    if "맛집" in content:
                        markdown.append("## 제천 맛집\n\n")
                    markdown.append(content)
                    markdown.append("\n\n")

        # Part 4: 숙박 (Page 22)
        markdown.append("---\n\n# PART 4: 편안한 휴식과 숙소\n\n")

        if 22 in pages:
            markdown.append("## 주요 숙박시설\n\n")
            markdown.append(self.clean_page_content(pages[22]))
            markdown.append("\n\n")

        # Part 5: 함께하는 제천 (Pages 23-28)
        markdown.append("---\n\n# PART 5: 함께하는 제천\n\n")

        for page_num in range(23, 29):
            if page_num in pages:
                content = self.clean_page_content(pages[page_num])
                if content:
                    if "기부제" in content:
                        markdown.append("## 고향사랑 기부제\n\n")
                    elif "꿀팁" in content:
                        markdown.append("## 알아두면 좋은 꿀팁\n\n")

                    markdown.append(content)
                    markdown.append("\n\n")

        return ''.join(markdown)

    def create_clean_markdown(self, output_path: str):
        """Main workflow to create clean markdown."""
        print("\n🚀 Creating Clean Markdown from PyMuPDF...\n")

        # Read file
        self.read_txt()

        # Extract pages
        pages = self.extract_pages()

        # Structure content
        markdown_content = self.structure_content(pages)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"\n✅ Clean markdown saved to: {output_path}")
        print(f"   Total length: {len(markdown_content)} chars")
        print(f"   Lines: {len(markdown_content.splitlines())}")

        # Compare with original
        original_md = Path("/home/user/goodganglabs/data/processed/제천시관광정보책자.md")
        if original_md.exists():
            with open(original_md, 'r', encoding='utf-8') as f:
                original_content = f.read()

            print(f"\n📊 Comparison:")
            print(f"   Original MD (pyzerox): {len(original_content)} chars")
            print(f"   Clean MD (PyMuPDF): {len(markdown_content)} chars")

            if len(markdown_content) < len(original_content):
                print(f"   ⚠️  Clean version is shorter by {len(original_content) - len(markdown_content)} chars")
                print(f"   Consider hybrid approach (pyzerox for images + PyMuPDF for accuracy)")
            else:
                print(f"   ✅ Clean version covers the content")


def main():
    """Main execution function."""
    # Paths
    txt_path = "/home/user/goodganglabs/data/processed/full_extraction/pymupdf_full.txt"
    output_path = "/home/user/goodganglabs/data/processed/제천시관광정보책자_clean.md"

    # Create clean markdown
    creator = CleanMarkdownCreator(txt_path)
    creator.create_clean_markdown(output_path)

    print("\n✅ Clean markdown creation completed!")
    print("\n💡 Recommendation:")
    print("   Use this clean version for RAG dataset generation")
    print("   PyMuPDF provides more accurate text extraction than OCR")


if __name__ == "__main__":
    main()
