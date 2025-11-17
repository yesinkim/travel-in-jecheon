"""
Complete Dataset Generation Pipeline

This script runs the complete pipeline to generate the Jecheon RAG dataset:
1. Extract document chunks
2. Generate Q&A pairs (requires ANTHROPIC_API_KEY)
3. Add distractor documents
4. Format training data
5. Split into train/test sets

Usage:
    export ANTHROPIC_API_KEY='your-api-key'
    python scripts/run_pipeline.py
"""

import os
import sys
import subprocess
from pathlib import Path


class PipelineRunner:
    """Runs the complete dataset generation pipeline."""

    SCRIPTS = [
        ("01_extract_pdf_chunks.py", "Document Chunking"),
        ("02_generate_qa_with_claude.py", "Q&A Generation (Claude API)"),
        ("03_add_distractors.py", "Distractor Addition"),
        ("04_format_training_data.py", "Training Data Formatting"),
        ("05_split_train_test.py", "Train/Test Split"),
    ]

    def __init__(self, skip_qa_generation: bool = False):
        """
        Initialize pipeline runner.

        Args:
            skip_qa_generation: If True, skip Q&A generation step (useful if already done)
        """
        self.skip_qa_generation = skip_qa_generation
        self.scripts_dir = Path("/home/user/goodganglabs/scripts")

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("🔍 Checking prerequisites...")

        # Check if running Q&A generation
        if not self.skip_qa_generation:
            # Check for API key
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("\n❌ Error: ANTHROPIC_API_KEY environment variable not set!")
                print("Please set it using: export ANTHROPIC_API_KEY='your-api-key'")
                print("\nOr run with --skip-qa-generation if Q&A pairs already exist.")
                return False
            else:
                print("✅ ANTHROPIC_API_KEY found")

        # Check if all scripts exist
        for script_name, _ in self.SCRIPTS:
            script_path = self.scripts_dir / script_name
            if not script_path.exists():
                print(f"❌ Script not found: {script_path}")
                return False

        print("✅ All scripts found")

        return True

    def run_script(self, script_name: str, description: str) -> bool:
        """
        Run a single script in the pipeline.

        Args:
            script_name: Name of the script file
            description: Human-readable description

        Returns:
            True if successful, False otherwise
        """
        # Skip Q&A generation if requested
        if self.skip_qa_generation and "02_generate_qa" in script_name:
            print(f"\n⏭️  Skipping: {description}")
            return True

        print(f"\n{'='*60}")
        print(f"▶️  Running: {description}")
        print(f"{'='*60}\n")

        script_path = self.scripts_dir / script_name

        try:
            result = subprocess.run(
                ["python", str(script_path)],
                check=True,
                capture_output=False,
                text=True
            )
            print(f"\n✅ Completed: {description}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error in {description}")
            print(f"Exit code: {e.returncode}")
            return False

        except Exception as e:
            print(f"\n❌ Unexpected error in {description}: {e}")
            return False

    def run_pipeline(self) -> bool:
        """
        Run the complete pipeline.

        Returns:
            True if all steps successful, False otherwise
        """
        print("\n" + "="*60)
        print("🚀 Starting Jecheon RAG Dataset Generation Pipeline")
        print("="*60)

        # Check prerequisites
        if not self.check_prerequisites():
            return False

        # Run each script in order
        for script_name, description in self.SCRIPTS:
            success = self.run_script(script_name, description)
            if not success:
                print(f"\n❌ Pipeline failed at: {description}")
                return False

        # Success!
        print("\n" + "="*60)
        print("✅ Pipeline completed successfully!")
        print("="*60)

        self.print_output_summary()

        return True

    def print_output_summary(self):
        """Print summary of generated files."""
        print("\n📁 Generated Files:")

        files = [
            "data/chunks/documents.jsonl",
            "data/chunks/qa_pairs.jsonl",
            "data/chunks/qa_with_distractors.jsonl",
            "data/processed/training_data.jsonl",
            "data/processed/dataset_hf.jsonl",
            "data/processed/train.jsonl",
            "data/processed/test.jsonl",
        ]

        for file_path in files:
            full_path = Path("/home/user/goodganglabs") / file_path
            if full_path.exists():
                size_kb = full_path.stat().st_size / 1024
                print(f"  ✅ {file_path} ({size_kb:.1f} KB)")
            else:
                print(f"  ❌ {file_path} (not found)")

        print("\n📊 Next Steps:")
        print("  1. Review the generated data for quality")
        print("  2. Upload dataset_hf.jsonl to Hugging Face Datasets")
        print("  3. Use train.jsonl for model fine-tuning")
        print("  4. Use test.jsonl for evaluation")


def main():
    """Main execution function."""
    # Parse command line arguments
    skip_qa_generation = "--skip-qa-generation" in sys.argv

    # Run pipeline
    runner = PipelineRunner(skip_qa_generation=skip_qa_generation)
    success = runner.run_pipeline()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
