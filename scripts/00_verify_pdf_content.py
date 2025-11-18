"""
PDF Content Verification and Enhancement Script

Compares pyzerox MD output with PyMuPDF TXT output to find and fill missing content.

Input:
- data/processed/제천시관광정보책자.md (pyzerox output)
- data/processed/full_extraction/pymupdf_full.txt (PyMuPDF output)

Output:
- data/processed/제천시관광정보책자_enhanced.md (enhanced version)
- data/processed/missing_content_report.txt (report of missing content)
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from difflib import SequenceMatcher


class PDFContentVerifier:
    """Verifies and enhances PDF extraction by comparing multiple sources."""

    def __init__(self, md_path: str, txt_path: str):
        """Initialize verifier with file paths."""
        self.md_path = Path(md_path)
        self.txt_path = Path(txt_path)
        self.md_content = ""
        self.txt_content = ""
        self.missing_sections = []

    def read_files(self):
        """Read both input files."""
        with open(self.md_path, 'r', encoding='utf-8') as f:
            self.md_content = f.read()

        with open(self.txt_path, 'r', encoding='utf-8') as f:
            self.txt_content = f.read()

        print(f"✅ Read MD file: {len(self.md_content)} chars")
        print(f"✅ Read TXT file: {len(self.txt_content)} chars")

    def clean_pymupdf_content(self) -> str:
        """Clean PyMuPDF output by removing headers and page markers."""
        lines = self.txt_content.split('\n')
        cleaned_lines = []
        skip_next = False

        for i, line in enumerate(lines):
            # Skip header lines
            if line.startswith("Parser:") or line.startswith("Pages:") or \
               line.startswith("Total characters:") or line.startswith("Time:") or \
               line.startswith("="*40):
                continue

            # Skip page markers
            if line.startswith("Page ") and i < len(lines) - 1 and lines[i+1].startswith("="*40):
                skip_next = True
                continue

            if skip_next:
                skip_next = False
                continue

            # Keep content lines
            if line.strip():
                cleaned_lines.append(line)

        cleaned = '\n'.join(cleaned_lines)
        print(f"✅ Cleaned TXT: {len(cleaned)} chars")
        return cleaned

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (remove extra whitespace, etc)."""
        # Remove line numbers from MD
        text = re.sub(r'^\s*\d+→', '', text, flags=re.MULTILINE)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special markers
        text = re.sub(r'[→•·]', '', text)
        return text.strip()

    def find_missing_sections(self, cleaned_txt: str) -> List[Dict]:
        """Find sections present in TXT but missing in MD."""
        # Split into sections (by common patterns)
        txt_sections = self.split_into_sections(cleaned_txt)
        md_sections = self.split_into_sections(self.md_content)

        missing = []

        for txt_section in txt_sections:
            # Check if this section exists in MD
            found = False
            txt_normalized = self.normalize_text(txt_section['content'])

            for md_section in md_sections:
                md_normalized = self.normalize_text(md_section['content'])

                # Use fuzzy matching
                similarity = SequenceMatcher(None, txt_normalized[:200], md_normalized[:200]).ratio()

                if similarity > 0.7:  # 70% similar
                    found = True
                    break

            if not found and len(txt_section['content'].strip()) > 100:
                missing.append({
                    'title': txt_section['title'],
                    'content': txt_section['content'],
                    'length': len(txt_section['content'])
                })

        return missing

    def split_into_sections(self, content: str) -> List[Dict]:
        """Split content into logical sections."""
        sections = []

        # Split by major headings or patterns
        patterns = [
            r'PART\s+\d+',
            r'디지털.*?관광주민증',
            r'제천.*?시티투어',
            r'관광택시',
            r'의림지',
            r'청풍',
            r'맛집',
            r'숙박',
            r'트레킹',
            r'코스',
        ]

        # Find section boundaries
        boundaries = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                boundaries.append((match.start(), match.group(0)))

        # Sort by position
        boundaries.sort(key=lambda x: x[0])

        # Extract sections
        for i, (start, title) in enumerate(boundaries):
            if i < len(boundaries) - 1:
                end = boundaries[i + 1][0]
            else:
                end = len(content)

            section_content = content[start:end].strip()
            if len(section_content) > 50:
                sections.append({
                    'title': title,
                    'content': section_content,
                    'start': start,
                    'end': end
                })

        return sections

    def generate_report(self, missing: List[Dict], output_path: str):
        """Generate a report of missing content."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Missing Content Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total missing sections: {len(missing)}\n\n")

            for i, section in enumerate(missing, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"Missing Section {i}: {section['title']}\n")
                f.write(f"Length: {section['length']} chars\n")
                f.write(f"{'='*60}\n\n")
                f.write(section['content'][:500] + "...\n")

        print(f"\n📄 Report saved to: {output_path}")

    def enhance_md_content(self, missing: List[Dict]) -> str:
        """Add missing content to MD file in appropriate locations."""
        enhanced = self.md_content

        print(f"\n🔧 Adding {len(missing)} missing sections...")

        for section in missing:
            # Try to find the best insertion point
            insertion_point = self.find_insertion_point(enhanced, section)

            if insertion_point:
                # Insert the missing content
                before = enhanced[:insertion_point]
                after = enhanced[insertion_point:]
                enhanced = f"{before}\n\n## {section['title']}\n\n{section['content']}\n\n{after}"
                print(f"  ✅ Added: {section['title']} ({section['length']} chars)")
            else:
                # Append at the end
                enhanced += f"\n\n## {section['title']}\n\n{section['content']}\n"
                print(f"  ⚠️  Appended to end: {section['title']}")

        return enhanced

    def find_insertion_point(self, content: str, section: Dict) -> int:
        """Find the best position to insert missing content."""
        # Look for similar heading or nearby context
        title_words = section['title'].split()

        for word in title_words:
            if len(word) > 2:  # Skip short words
                # Find where this word appears in content
                pattern = re.escape(word)
                matches = list(re.finditer(pattern, content, re.IGNORECASE))

                if matches:
                    # Return position after the last match
                    return matches[-1].end()

        return None

    def verify_and_enhance(self):
        """Main verification and enhancement workflow."""
        print("\n🚀 Starting PDF Content Verification...\n")

        # Read files
        self.read_files()

        # Clean PyMuPDF content
        cleaned_txt = self.clean_pymupdf_content()

        # Find missing sections
        print("\n🔍 Searching for missing content...")
        missing = self.find_missing_sections(cleaned_txt)

        print(f"\n📊 Found {len(missing)} potentially missing sections")

        if missing:
            # Show summary
            print("\n📋 Missing Sections Summary:")
            for i, section in enumerate(missing, 1):
                print(f"  {i}. {section['title'][:50]}... ({section['length']} chars)")

            # Generate report
            report_path = self.md_path.parent / "missing_content_report.txt"
            self.generate_report(missing, report_path)

            # Enhance MD content
            enhanced_content = self.enhance_md_content(missing)

            # Save enhanced version
            enhanced_path = self.md_path.parent / "제천시관광정보책자_enhanced.md"
            with open(enhanced_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)

            print(f"\n✅ Enhanced MD saved to: {enhanced_path}")
            print(f"   Original: {len(self.md_content)} chars")
            print(f"   Enhanced: {len(enhanced_content)} chars")
            print(f"   Added: {len(enhanced_content) - len(self.md_content)} chars")

        else:
            print("\n✅ No missing content found! MD file is complete.")

        print("\n✅ Verification completed!")


def main():
    """Main execution function."""
    # Paths
    md_path = "/home/user/goodganglabs/data/processed/제천시관광정보책자.md"
    txt_path = "/home/user/goodganglabs/data/processed/full_extraction/pymupdf_full.txt"

    # Initialize verifier
    verifier = PDFContentVerifier(md_path, txt_path)

    # Run verification and enhancement
    verifier.verify_and_enhance()


if __name__ == "__main__":
    main()
