"""Compare different PDF parsing methods including OCR."""

import sys
import time
import os
import asyncio
from pathlib import Path
from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def parse_with_pymupdf(pdf_path: str) -> Dict[str, Any]:
    """Parse PDF using PyMuPDF (fitz)."""
    import pymupdf

    start_time = time.time()
    text = ""
    page_count = 0

    try:
        doc = pymupdf.open(pdf_path)
        page_count = len(doc)

        for page in doc:
            text += page.get_text()

        doc.close()

        elapsed = time.time() - start_time

        return {
            "name": "PyMuPDF (fitz)",
            "success": True,
            "text": text,
            "page_count": page_count,
            "char_count": len(text),
            "elapsed_time": elapsed,
            "speed": page_count / elapsed if elapsed > 0 else 0,
            "description": "Fast, accurate, good for standard PDFs"
        }
    except Exception as e:
        return {
            "name": "PyMuPDF (fitz)",
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }


def parse_with_pdfplumber(pdf_path: str) -> Dict[str, Any]:
    """Parse PDF using pdfplumber."""
    import pdfplumber

    start_time = time.time()
    text = ""
    page_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)

            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        elapsed = time.time() - start_time

        return {
            "name": "pdfplumber",
            "success": True,
            "text": text,
            "page_count": page_count,
            "char_count": len(text),
            "elapsed_time": elapsed,
            "speed": page_count / elapsed if elapsed > 0 else 0,
            "description": "Good for tables and layout preservation"
        }
    except Exception as e:
        return {
            "name": "pdfplumber",
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }


def parse_with_pypdf2(pdf_path: str) -> Dict[str, Any]:
    """Parse PDF using PyPDF2."""
    from PyPDF2 import PdfReader

    start_time = time.time()
    text = ""
    page_count = 0

    try:
        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)

        for page in reader.pages:
            text += page.extract_text() + "\n"

        elapsed = time.time() - start_time

        return {
            "name": "PyPDF2",
            "success": True,
            "text": text,
            "page_count": page_count,
            "char_count": len(text),
            "elapsed_time": elapsed,
            "speed": page_count / elapsed if elapsed > 0 else 0,
            "description": "Simple, lightweight, basic text extraction"
        }
    except Exception as e:
        return {
            "name": "PyPDF2",
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }


def parse_with_pymupdf_ocr(pdf_path: str, max_pages: int = 3) -> Dict[str, Any]:
    """Parse PDF using PyMuPDF with OCR for images."""
    import pymupdf

    start_time = time.time()
    text = ""
    page_count = 0
    images_found = 0

    try:
        doc = pymupdf.open(pdf_path)
        # Limit to first few pages for OCR (it's slow)
        pages_to_process = min(len(doc), max_pages)
        page_count = len(doc)

        for page_num in range(pages_to_process):
            page = doc[page_num]

            # Get text normally
            page_text = page.get_text()
            text += page_text + "\n"

            # Extract images for potential OCR
            image_list = page.get_images()
            images_found += len(image_list)

        doc.close()

        elapsed = time.time() - start_time

        return {
            "name": f"PyMuPDF + Image Detection (first {pages_to_process} pages)",
            "success": True,
            "text": text,
            "page_count": page_count,
            "pages_processed": pages_to_process,
            "char_count": len(text),
            "images_found": images_found,
            "elapsed_time": elapsed,
            "speed": pages_to_process / elapsed if elapsed > 0 else 0,
            "description": "Text extraction + image detection (OCR-ready)"
        }
    except Exception as e:
        return {
            "name": "PyMuPDF + OCR",
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }


def parse_with_ocr(pdf_path: str, max_pages: int = 2) -> Dict[str, Any]:
    """Parse PDF using OCR (pytesseract + pdf2image)."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        return {
            "name": "OCR (pytesseract)",
            "success": False,
            "error": f"Missing dependency: {e}. Install poppler-utils and tesseract."
        }

    start_time = time.time()
    text = ""

    try:
        # Convert PDF to images (limit pages for performance)
        images = convert_from_path(pdf_path, first_page=1, last_page=max_pages)

        # OCR each image
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image, lang='kor+eng')
            text += f"\n--- Page {i+1} ---\n{page_text}\n"

        elapsed = time.time() - start_time

        return {
            "name": f"OCR (pytesseract, first {max_pages} pages)",
            "success": True,
            "text": text,
            "page_count": len(images),
            "char_count": len(text),
            "elapsed_time": elapsed,
            "speed": len(images) / elapsed if elapsed > 0 else 0,
            "description": "OCR for scanned PDFs (requires poppler & tesseract)",
            "note": "⚠️ Requires: brew install poppler tesseract tesseract-lang"
        }
    except Exception as e:
        return {
            "name": "OCR (pytesseract)",
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "note": "⚠️ Install: brew install poppler tesseract tesseract-lang"
        }


async def parse_with_zerox_async(pdf_path: str, max_pages: int = 5) -> Dict[str, Any]:
    """Parse PDF using py-zerox (Vision LLM based)."""
    try:
        from pyzerox import zerox
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError as e:
        return {
            "name": "py-zerox (Vision LLM)",
            "success": False,
            "error": f"Missing dependency: {e}",
            "note": "⚠️ Install: uv pip install py-zerox python-dotenv"
        }

    start_time = time.time()

    try:
        # Check for API key (priority: GEMINI > OpenAI > Anthropic)
        api_key = None
        model = None

        # Gemini can use either GEMINI_API_KEY or GOOGLE_API_KEY
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        # if gemini_key:
        #     # Set env var for litellm
        #     os.environ["GEMINI_API_KEY"] = gemini_key
        #     # Use latest Gemini Pro model (supports vision)
        #     model = "gemini/gemini-2.5-pro"
        #     print(f"Using Google Gemini 1.5 Pro model")
        #     print(f"API Key found: {gemini_key[:10]}...")
        if os.getenv("OPENAI_API_KEY"):
            api_key = os.getenv("OPENAI_API_KEY")
            model = "gpt-4o-mini"
            print(f"Using OpenAI model: {model}")
        else:
            return {
                "name": "py-zerox (Vision LLM)",
                "success": False,
                "error": "No API key found",
                "note": "⚠️ Set GEMINI_API_KEY (or GOOGLE_API_KEY), OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env"
            }

        # Extract with zerox (limit pages to save costs)
        result = await zerox(
            file_path=pdf_path,
            model=model,
            cleanup=True,
            api_key=api_key,
        )

        elapsed = time.time() - start_time

        # Combine text from first N pages
        text = ""
        pages_to_use = min(len(result.pages), max_pages)
        for i in range(pages_to_use):
            text += result.pages[i].content + "\n"

        # Get token count if available
        tokens = getattr(result, 'completion_tokens', 'N/A')
        tokens_str = f"{tokens:,}" if isinstance(tokens, int) else str(tokens)

        return {
            "name": f"py-zerox (first {pages_to_use} pages)",
            "success": True,
            "text": text,
            "page_count": len(result.pages),
            "pages_processed": pages_to_use,
            "char_count": len(text),
            "elapsed_time": elapsed,
            "speed": pages_to_use / elapsed if elapsed > 0 else 0,
            "description": f"Vision LLM extraction ({model})",
            "note": f"⚠️ Uses API credits. Tokens: {tokens_str}"
        }

    except Exception as e:
        return {
            "name": "py-zerox (Vision LLM)",
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "note": "⚠️ Requires API key in .env file"
        }


def parse_with_zerox(pdf_path: str, max_pages: int = 5) -> Dict[str, Any]:
    """Wrapper to run async zerox parser."""
    return asyncio.run(parse_with_zerox_async(pdf_path, max_pages))


def save_sample_output(result: Dict[str, Any], output_dir: Path):
    """Save sample output from each parser."""
    if not result["success"] or "text" not in result:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save first 2000 characters as sample
    sample_text = result["text"][:2000]
    filename = output_dir / f"{result['name'].replace(' ', '_').replace('/', '_')}_sample.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Parser: {result['name']}\n")
        f.write(f"Description: {result['description']}\n")
        f.write(f"Total characters: {result['char_count']}\n")
        f.write(f"Time: {result['elapsed_time']:.2f}s\n")
        f.write("=" * 60 + "\n\n")
        f.write(sample_text)


def compare_parsers(pdf_path: str, output_dir: str = "data/processed/parser_comparison"):
    """Compare all PDF parsing methods."""
    print("=" * 80)
    print("PDF Parser Comparison")
    print("=" * 80)
    print(f"\nPDF: {pdf_path}\n")

    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return

    output_path = Path(output_dir)

    # List of parsers to test
    parsers = [
        # ("PyMuPDF", parse_with_pymupdf),
        # ("pdfplumber", parse_with_pdfplumber),
        # ("PyPDF2", parse_with_pypdf2),
        # ("PyMuPDF + Images", parse_with_pymupdf_ocr),
        # ("OCR", parse_with_ocr),
        ("py-zerox", parse_with_zerox),
    ]

    results = []

    for parser_name, parser_func in parsers:
        print(f"\n[Testing {parser_name}]")
        print("-" * 80)

        result = parser_func(pdf_path)
        results.append(result)

        if result["success"]:
            print(f"✓ {result['name']}")
            print(f"  Pages: {result.get('page_count', 'N/A')}")
            print(f"  Characters extracted: {result['char_count']:,}")
            print(f"  Time: {result['elapsed_time']:.2f}s")
            print(f"  Speed: {result['speed']:.2f} pages/sec")
            print(f"  Description: {result['description']}")

            if "note" in result:
                print(f"  {result['note']}")

            # Save sample
            save_sample_output(result, output_path)
        else:
            print(f"❌ {result['name']} failed")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            if "note" in result:
                print(f"  {result['note']}")

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    successful_results = [r for r in results if r["success"]]

    if successful_results:
        print(f"\n{'Parser':<40} {'Chars':<12} {'Time':<10} {'Speed'}")
        print("-" * 80)

        for result in successful_results:
            name = result["name"][:38]
            chars = f"{result['char_count']:,}"
            time_str = f"{result['elapsed_time']:.2f}s"
            speed = f"{result['speed']:.2f} p/s"
            print(f"{name:<40} {chars:<12} {time_str:<10} {speed}")

        # Best performer
        fastest = max(successful_results, key=lambda x: x["speed"])
        most_text = max(successful_results, key=lambda x: x["char_count"])

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        print(f"\n⚡ Fastest: {fastest['name']}")
        print(f"   {fastest['speed']:.2f} pages/sec")
        print(f"\n📄 Most text extracted: {most_text['name']}")
        print(f"   {most_text['char_count']:,} characters")

        print("\n💡 For Korean tourism PDF:")
        print("   - If PDF has selectable text: PyMuPDF (fastest, most reliable)")
        print("   - If PDF has tables: pdfplumber")
        print("   - If PDF is scanned/image: OCR (requires external tools)")

        print(f"\n📁 Sample outputs saved to: {output_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "data/raw/제천시관광정보책자.pdf"

    compare_parsers(pdf_path)
