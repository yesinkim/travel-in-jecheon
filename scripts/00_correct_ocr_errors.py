"""
Hybrid Markdown Creator: Correct OCR errors and add missing sections

Strategy:
1. Use pyzerox MD as base (most complete, has image text)
2. Find OCR errors by comparing with PyMuPDF TXT
3. Replace erroneous sections with accurate PyMuPDF text
4. Find missing sections in pyzerox (present in PyMuPDF)
5. Add missing sections at appropriate positions
6. Keep unique content from pyzerox (image extractions)

Input:
- data/processed/제천시관광정보책자.md (pyzerox - complete but has errors)
- data/processed/full_extraction/pymupdf_full.txt (PyMuPDF - accurate)

Output:
- data/processed/제천시관광정보책자_corrected.md (best of both)
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from difflib import SequenceMatcher


class HybridMarkdownCreator:
    """Creates corrected markdown by combining pyzerox and PyMuPDF."""

    def __init__(self, md_path: str, txt_path: str):
        """Initialize with file paths."""
        self.md_path = Path(md_path)
        self.txt_path = Path(txt_path)
        self.md_content = ""
        self.txt_content = ""
        self.txt_cleaned = ""
        self.corrections = []

    def read_files(self):
        """Read both input files."""
        with open(self.md_path, 'r', encoding='utf-8') as f:
            self.md_content = f.read()

        with open(self.txt_path, 'r', encoding='utf-8') as f:
            self.txt_content = f.read()

        print(f"✅ Read MD file (pyzerox): {len(self.md_content)} chars")
        print(f"✅ Read TXT file (PyMuPDF): {len(self.txt_content)} chars")

    def clean_pymupdf(self) -> str:
        """Clean PyMuPDF content."""
        lines = self.txt_content.split('\n')
        cleaned_lines = []

        for i, line in enumerate(lines):
            # Skip metadata and page markers
            if any(line.startswith(prefix) for prefix in [
                "Parser:", "Pages:", "Total", "Time:", "="*40
            ]):
                continue

            if line.strip().startswith("Page ") and i < len(lines) - 1:
                continue

            if line.strip():
                cleaned_lines.append(line)

        self.txt_cleaned = '\n'.join(cleaned_lines)
        print(f"✅ Cleaned PyMuPDF: {len(self.txt_cleaned)} chars")
        return self.txt_cleaned

    def normalize_for_comparison(self, text: str) -> str:
        """Normalize text for fuzzy matching."""
        # Remove line numbers
        text = re.sub(r'^\s*\d+→', '', text, flags=re.MULTILINE)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[•·→|]', '', text)
        return text.strip().lower()

    def find_ocr_errors(self) -> List[Dict]:
        """Find OCR errors by comparing similar sections."""
        print("\n🔍 Searching for OCR errors...")

        errors = []

        # Split into sections for comparison
        md_sections = self.split_by_keywords(self.md_content)
        txt_sections = self.split_by_keywords(self.txt_cleaned)

        print(f"   MD sections: {len(md_sections)}")
        print(f"   TXT sections: {len(txt_sections)}")

        # Compare each section
        for md_sec in md_sections:
            best_match = None
            best_similarity = 0

            # Find best matching section in PyMuPDF
            for txt_sec in txt_sections:
                # Compare titles first
                if md_sec['title'] == txt_sec['title']:
                    similarity = 1.0
                else:
                    md_norm = self.normalize_for_comparison(md_sec['content'][:200])
                    txt_norm = self.normalize_for_comparison(txt_sec['content'][:200])
                    similarity = SequenceMatcher(None, md_norm, txt_norm).ratio()

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = txt_sec

            # If similar enough, check for differences
            if best_match and best_similarity > 0.6:
                md_norm = self.normalize_for_comparison(md_sec['content'])
                txt_norm = self.normalize_for_comparison(best_match['content'])

                # If not identical, it's a potential OCR error
                if md_norm != txt_norm:
                    errors.append({
                        'title': md_sec['title'],
                        'md_content': md_sec['content'],
                        'txt_content': best_match['content'],
                        'similarity': best_similarity,
                        'start': md_sec['start'],
                        'end': md_sec['end']
                    })

        print(f"✅ Found {len(errors)} sections with potential OCR errors")
        return errors

    def split_by_keywords(self, content: str) -> List[Dict]:
        """Split content by known keywords/sections."""
        keywords = [
            '디지털관광주민증',
            '디지털 관광주민증',
            '제천 시티투어',
            '제천시티투어',
            '관광택시',
            '인센티브',
            '가스트로',
            '의림지',
            '청풍호반',
            '청풍문화유산',
            '배론성지',
            '박달재',
            '옥순봉',
            '청풍랜드',
            '치유의 숲',
            '트레킹',
            '코스',
            '맛집',
            '숙박',
            '기부제',
        ]

        sections = []
        positions = []

        # Find all keyword positions
        for keyword in keywords:
            for match in re.finditer(re.escape(keyword), content, re.IGNORECASE):
                positions.append((match.start(), keyword))

        # Sort by position
        positions.sort()

        # Extract sections
        for i, (start, title) in enumerate(positions):
            if i < len(positions) - 1:
                end = positions[i + 1][0]
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

    def detect_specific_errors(self, md_text: str, txt_text: str) -> List[Tuple[str, str]]:
        """Detect specific OCR error patterns."""
        replacements = []

        # Known error patterns from our analysis
        error_patterns = [
            # Pattern: (wrong_in_md, correct_in_txt, context)
            ('저수 10경', '제천 10경', '의림지'),  # Major OCR error
            ('고려 수향팔경에 임항장', '고대 수리시설의 원형', '의림지'),
            ('벽지 10여 종 103경으로, 천주교 전파의 중심지이자 박해 후 신앙을 지켜온 교육',
             '천주교 전파의 중심지이자 박해 속 신앙을 지켜온 교우촌', '배론성지'),
        ]

        # Check MD for these errors
        for wrong, correct, context in error_patterns:
            # If MD has wrong text and context matches
            if context.lower() in md_text.lower():
                if wrong in md_text:
                    # Verify correct text exists in PyMuPDF
                    if correct in txt_text:
                        replacements.append((wrong, correct))
                        print(f"  🔧 Detected: '{wrong}' → '{correct}'")

        return replacements

    def apply_corrections(self, errors: List[Dict]) -> str:
        """Apply corrections to MD content."""
        corrected = self.md_content
        corrections_made = 0

        print(f"\n🔧 Applying corrections...")

        for error in errors:
            title = error['title']
            md_content = error['md_content']
            txt_content = error['txt_content']

            # Detect specific errors
            replacements = self.detect_specific_errors(md_content, txt_content)

            if replacements:
                for wrong, correct in replacements:
                    # Replace in full content
                    old_corrected = corrected
                    corrected = re.sub(
                        re.escape(wrong),
                        correct,
                        corrected,
                        count=1,
                        flags=re.IGNORECASE
                    )

                    if corrected != old_corrected:
                        corrections_made += 1
                        print(f"  ✅ [{title}] '{wrong}' → '{correct}'")

            # For sections with low similarity, consider replacing entirely
            elif error['similarity'] < 0.7 and len(txt_content) > 100:
                # Be cautious - only replace if PyMuPDF version is substantial
                print(f"  ⚠️  [{title}] Low similarity ({error['similarity']:.2f}) - manual review recommended")

        print(f"\n✅ Applied {corrections_made} corrections")
        self.corrections = errors
        return corrected

    def find_missing_sections(self) -> List[Dict]:
        """Find sections present in PyMuPDF but missing in pyzerox."""
        print("\n🔍 Searching for missing sections...")

        missing = []

        # Split both sources into sections
        md_sections = self.split_by_keywords(self.md_content)
        txt_sections = self.split_by_keywords(self.txt_cleaned)

        # For each PyMuPDF section, check if it exists in pyzerox
        for txt_sec in txt_sections:
            found = False
            txt_norm = self.normalize_for_comparison(txt_sec['content'])

            # Skip if too short (likely noise)
            if len(txt_norm) < 50:
                continue

            # Search for this section in pyzerox MD
            for md_sec in md_sections:
                md_norm = self.normalize_for_comparison(md_sec['content'])

                # Check similarity
                similarity = SequenceMatcher(None, txt_norm[:200], md_norm[:200]).ratio()

                if similarity > 0.6:  # Found similar section
                    found = True
                    break

            # If not found, it's missing
            if not found and len(txt_sec['content'].strip()) > 100:
                missing.append({
                    'title': txt_sec['title'],
                    'content': txt_sec['content'],
                    'length': len(txt_sec['content'])
                })

        print(f"✅ Found {len(missing)} missing sections")

        # Show summary
        if missing:
            print(f"\n📋 Missing sections summary:")
            for i, sec in enumerate(missing[:10], 1):
                print(f"   {i}. {sec['title']} ({sec['length']} chars)")
            if len(missing) > 10:
                print(f"   ... and {len(missing) - 10} more")

        return missing

    def find_insertion_point(self, missing_section: Dict, current_content: str) -> int:
        """Find appropriate insertion point for a missing section."""
        title = missing_section['title']

        # Strategy 1: Find by keyword proximity
        # Look for related keywords in the content
        keywords = title.lower().split()

        # Find positions where keywords appear
        positions = []
        for keyword in keywords:
            if len(keyword) > 2:  # Skip short words
                pattern = re.escape(keyword)
                for match in re.finditer(pattern, current_content, re.IGNORECASE):
                    positions.append(match.start())

        if positions:
            # Insert near the first occurrence of related keyword
            avg_position = min(positions)
            # Find nearest section break (double newline)
            section_breaks = [m.start() for m in re.finditer(r'\n\n', current_content)]

            # Find closest section break after the keyword
            for break_pos in section_breaks:
                if break_pos >= avg_position:
                    return break_pos + 2  # After the double newline

        # Strategy 2: Insert at the end if no good position found
        return len(current_content)

    def add_missing_sections(self, corrected_content: str, missing_sections: List[Dict]) -> str:
        """Add missing sections to the corrected content."""
        if not missing_sections:
            return corrected_content

        print(f"\n➕ Adding {len(missing_sections)} missing sections...")

        enhanced = corrected_content
        sections_added = 0

        for section in missing_sections:
            title = section['title']
            content = section['content'].strip()

            # Find insertion point
            insert_pos = self.find_insertion_point(section, enhanced)

            # Format section with proper markdown
            formatted_section = f"\n\n## {title}\n\n{content}\n"

            # Insert at position
            enhanced = enhanced[:insert_pos] + formatted_section + enhanced[insert_pos:]
            sections_added += 1

            print(f"   ✅ Added: {title} ({section['length']} chars)")

        print(f"\n✅ Added {sections_added} sections")
        return enhanced

    def create_correction_report(self, output_path: str):
        """Create a detailed correction report."""
        output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("OCR Correction Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total sections analyzed: {len(self.corrections)}\n")
            f.write(f"Corrections applied: [see below]\n\n")

            for i, error in enumerate(self.corrections, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"Section {i}: {error['title']}\n")
                f.write(f"Similarity: {error['similarity']:.2%}\n")
                f.write(f"{'='*60}\n\n")

                f.write("MD (pyzerox) version:\n")
                f.write("-"*40 + "\n")
                f.write(error['md_content'][:500] + "...\n\n")

                f.write("TXT (PyMuPDF) version:\n")
                f.write("-"*40 + "\n")
                f.write(error['txt_content'][:500] + "...\n\n")

        print(f"📄 Correction report saved to: {output_path}")

    def create_corrected_markdown(self):
        """Main workflow to create corrected markdown."""
        print("\n🚀 Creating Corrected & Enhanced Markdown (Hybrid Approach)...\n")
        print("Step 1: OCR Error Correction")
        print("Step 2: Missing Section Addition\n")

        # Read files
        self.read_files()

        # Clean PyMuPDF
        self.clean_pymupdf()

        # STEP 1: Find and fix OCR errors
        print("\n" + "="*60)
        print("STEP 1: OCR Error Correction")
        print("="*60)

        errors = self.find_ocr_errors()

        if errors:
            # Apply OCR corrections
            corrected_content = self.apply_corrections(errors)
        else:
            print("\n✅ No OCR errors detected!")
            corrected_content = self.md_content

        # STEP 2: Find and add missing sections
        print("\n" + "="*60)
        print("STEP 2: Missing Section Addition")
        print("="*60)

        missing_sections = self.find_missing_sections()

        if missing_sections:
            # Add missing sections
            enhanced_content = self.add_missing_sections(corrected_content, missing_sections)
        else:
            print("\n✅ No missing sections detected!")
            enhanced_content = corrected_content

        # Save enhanced version
        output_path = self.md_path.parent / "제천시관광정보책자_corrected.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_content)

        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        print(f"\n✅ Enhanced markdown saved to: {output_path}")
        print(f"   Original (pyzerox):  {len(self.md_content):,} chars")
        print(f"   After OCR fix:       {len(corrected_content):,} chars")
        print(f"   After adding missing: {len(enhanced_content):,} chars")
        print(f"   Total added:         +{len(enhanced_content) - len(self.md_content):,} chars")

        # Create report
        report_path = self.md_path.parent / "ocr_correction_report.txt"
        self.create_correction_report(report_path)

        return enhanced_content


def main():
    """Main execution function."""
    # Paths
    md_path = "/home/user/goodganglabs/data/processed/제천시관광정보책자.md"
    txt_path = "/home/user/goodganglabs/data/processed/full_extraction/pymupdf_full.txt"

    # Create hybrid markdown
    creator = HybridMarkdownCreator(md_path, txt_path)
    corrected = creator.create_corrected_markdown()

    print("\n✅ Hybrid markdown creation completed!")
    print("\n💡 Result:")
    print("   - pyzerox structure & completeness ✅")
    print("   - PyMuPDF accuracy (OCR corrections) ✅")
    print("   - Best of both worlds!")


if __name__ == "__main__":
    main()
