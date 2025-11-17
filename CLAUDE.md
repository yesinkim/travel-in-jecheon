# CLAUDE.md - AI Assistant Guide for GoodGangLabs RAG Project

## Project Overview

This repository contains an AI position assignment for **GoodGangLabs**, focusing on building a **RAG (Retrieval-Augmented Generation)** system with LLM fine-tuning capabilities.

### Project Goals
1. Build a RAG dataset from a provided Korean tourism PDF (Jecheon city information)
2. Implement and evaluate a baseline open-source LLM
3. Fine-tune the model for RAG tasks
4. Compare baseline vs fine-tuned performance
5. Upload dataset and model to Hugging Face
6. Create comprehensive documentation and analysis

### Current State
- **Phase:** Initial setup (no implementation code yet)
- **Status:** Documentation and assets in place, ready for development
- **Branch:** `claude/claude-md-mhyifajyydv2kwaf-01RitCh541mcZbfHbpV3LGYj`

---

## Repository Structure

```
/home/user/goodganglabs/
├── .git/                          # Git version control
├── .gitignore                     # Ignores: .DS_Store, large files
├── .python-version                # Python version specification
├── CLAUDE.md                      # This file - AI assistant guide
├── AI 포지션 과제 안내 2aa86229d43581a688b7c2ed89434dd5.md
│                                  # Assignment requirements (Korean)
├── README.md                      # Project README
├── main.py                        # Main entry point
├── pyproject.toml                 # Project dependencies (uv)
├── uv.lock                        # Dependency lock file
├── data/                          # Data directory
│   └── raw/                       # Raw source data
│       └── 제천시관광정보책자.pdf  # Source PDF for RAG (12MB tourism brochure)
├── docs/                          # Documentation files
│   └── Runpod설정방법.pdf         # RunPod GPU setup guide (15 pages)
└── notebook/                      # Jupyter notebooks
    └── test-models.ipynb          # Model testing notebook
```

### Expected Directory Structure (To Be Created)

```
/home/user/goodganglabs/
├── src/                           # Source code
│   ├── data_processing/           # PDF parsing, chunking, embedding
│   ├── rag_pipeline/              # RAG system implementation
│   ├── training/                  # Fine-tuning scripts
│   └── evaluation/                # Metrics and evaluation code
├── data/                          # Processed datasets (gitignored)
│   ├── raw/                       # Original data
│   ├── processed/                 # Cleaned and chunked data
│   ├── train/                     # Training dataset
│   └── test/                      # Evaluation dataset
├── notebooks/                     # Jupyter notebooks for exploration
├── scripts/                       # Standalone scripts
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── inference.py               # Inference script
├── configs/                       # Configuration files
│   ├── model_config.yaml          # Model hyperparameters
│   └── training_config.yaml       # Training settings
├── results/                       # Evaluation results (gitignored)
│   ├── metrics/                   # Performance metrics
│   ├── visualizations/            # Graphs and charts
│   └── examples/                  # Sample outputs
├── models/                        # Local model checkpoints (gitignored)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## Assignment Context

### Key Requirements

**Assignment Document:** `AI 포지션 과제 안내 2aa86229d43581a688b7c2ed89434dd5.md`

#### Deliverables Checklist
- [ ] Hugging Face Dataset link (RAG training/evaluation dataset)
- [ ] Hugging Face Model link (Fine-tuned model)
- [ ] Implementation code (GitHub repository)
- [ ] Results report (PDF or Markdown) including:
  - [ ] Data construction explanation
  - [ ] LLM and embedding model selection rationale
  - [ ] Baseline vs Fine-tuned performance comparison (with screenshots)
  - [ ] Performance evaluation analysis

#### Task Breakdown

1. **RAG Dataset Construction**
   - Parse and process `제천시관광정보책자.pdf` (Jecheon tourism brochure)
   - Create question-answer pairs or context-query pairs
   - Upload to Hugging Face Dataset
   - LLM usage (GPT, Claude) is permitted for data generation

2. **Baseline Model Evaluation**
   - Select open-source LLM (Llama, Qwen, Gemma, Phi, etc.)
   - Justify model selection
   - Choose and justify evaluation metrics
   - Run inference on test set
   - Document performance

3. **RAG Fine-tuning**
   - Fine-tune selected model on RAG task
   - Document training strategy and hyperparameters
   - Track loss curves and training metrics
   - Upload fine-tuned model to Hugging Face Model Hub

4. **Evaluation & Comparison**
   - Use same metrics as baseline
   - Compare baseline vs fine-tuned performance
   - Provide qualitative analysis (3+ example comparisons)
   - Analyze failure cases and limitations

#### Resources
- **GPU Platform:** RunPod (credit: `c1w0wzucm2khx7qk0br9`)
- **Setup Guide:** `docs/Runpod설정방법.pdf`
- **Reference Dataset:** https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO
- **Contact:** dasol@goodganglabs.com

---

## Development Workflow

### 1. Initial Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (after creating requirements.txt)
pip install -r requirements.txt

# Create necessary directories
mkdir -p src/{data_processing,rag_pipeline,training,evaluation}
mkdir -p data/{raw,processed,train,test}
mkdir -p notebooks scripts configs results models
```

### 2. Data Processing Pipeline

```
PDF → Text Extraction → Chunking → Embedding → Dataset Creation → HuggingFace Upload
```

**Key Considerations:**
- Text extraction from Korean PDF (handle encoding properly)
- Chunk size optimization for RAG
- Overlap strategy for context preservation
- Train/test split ratio
- Data quality validation

### 3. Model Development Pipeline

```
Model Selection → Baseline Evaluation → Fine-tuning → Fine-tuned Evaluation → Comparison
```

### 4. Git Workflow

**Branch Strategy:**
- **Working Branch:** `claude/claude-md-mhyifajyydv2kwaf-01RitCh541mcZbfHbpV3LGYj`
- All development happens on this branch
- Commit frequently with clear messages
- Push to remote when milestones are reached

**Commit Message Convention:**
```
<type>: <brief description>

<detailed explanation if needed>

Types: feat, fix, docs, refactor, test, chore
```

**Push Protocol:**
```bash
# Always use -u flag for branch tracking
git push -u origin claude/claude-md-mhyifajyydv2kwaf-01RitCh541mcZbfHbpV3LGYj

# On network failures, retry with exponential backoff (2s, 4s, 8s, 16s)
```

---

## Key Conventions for AI Assistants

### 1. Language Considerations

**Primary Language:** Korean (assignment and source data)
- Assignment document is in Korean
- Source PDF is Korean (Jecheon tourism information)
- Code comments should be in English for international standards
- Documentation can be bilingual (Korean + English) for accessibility
- Variable names in English following Python conventions

### 2. File Naming Conventions

- **Python files:** `snake_case.py`
- **Configuration files:** `lowercase_config.yaml`
- **Notebooks:** `01_descriptive_name.ipynb` (numbered for ordering)
- **Data files:** `descriptive_name_YYYYMMDD.json/csv/parquet`
- **Model checkpoints:** `model_name_epoch_step.pt`

### 3. Code Style Guidelines

**Python:**
- Follow PEP 8 standards
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings (Google or NumPy style)
- Import order: standard library → third-party → local

**Example:**
```python
from typing import List, Dict, Tuple
import torch
from transformers import AutoModel

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for RAG processing.

    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    # Implementation here
    pass
```

### 4. Configuration Management

- Store all hyperparameters in config files (YAML or JSON)
- Keep separate configs for development vs production
- Never hardcode API keys or credentials
- Use environment variables for sensitive data

### 5. Data Handling

- **Never commit large files** (>100MB) to git
- Use `.gitignore` for:
  - `/data/` (except small samples)
  - `/models/` (checkpoints)
  - `/results/`
  - `*.pt`, `*.pth`, `*.bin`
  - `.env` files
  - `__pycache__/`
- Document data preprocessing steps thoroughly
- Validate data quality at each stage

### 6. Experiment Tracking

**Recommended Approach:**
- Use Weights & Biases, MLflow, or TensorBoard
- Log all hyperparameters
- Track metrics during training
- Save model checkpoints at intervals
- Document experiment configurations

**Minimum Tracking:**
```python
experiment_metadata = {
    "model_name": "Qwen/Qwen2.5-7B",
    "dataset_version": "v1.0",
    "training_date": "2025-11-14",
    "hyperparameters": {
        "learning_rate": 2e-5,
        "batch_size": 4,
        "epochs": 3,
        "lora_r": 16,
        "lora_alpha": 32
    },
    "metrics": {
        "baseline_accuracy": 0.65,
        "finetuned_accuracy": 0.78
    }
}
```

### 7. Testing Strategy

**Unit Tests:**
- Test data processing functions
- Test chunking and embedding logic
- Test metric calculations

**Integration Tests:**
- Test end-to-end RAG pipeline
- Test inference workflow
- Validate output formats

**Evaluation Tests:**
- Ensure reproducibility of metrics
- Validate test/train data separation
- Check for data leakage

### 8. Documentation Requirements

**Code Documentation:**
- Docstrings for all public functions and classes
- Inline comments for complex logic
- Type hints for better IDE support

**Project Documentation:**
- Keep README.md updated with:
  - Setup instructions
  - Usage examples
  - Project structure
  - Results summary
- Document design decisions in separate docs or wiki
- Maintain CHANGELOG.md for significant updates

**Report Documentation (Required Deliverable):**
Must include 7 sections:
1. Project overview
2. RAG dataset construction process
3. Baseline model evaluation
4. Fine-tuning methodology
5. Fine-tuned model evaluation & comparison
6. Implementation code explanation
7. Conclusion & future improvements

### 9. Model Selection Guidance

**Factors to Consider:**
- Model size vs available GPU memory
- Korean language support quality
- License compatibility (Apache 2.0 preferred)
- Community support and documentation
- Inference speed requirements

**Recommended Models for Korean RAG:**
- **Qwen2.5-7B/14B** - Strong Korean support, good instruction following
- **Llama-3-8B** - Well-documented, good performance
- **Gemma-7B** - Efficient, good multilingual support
- **Phi-3** - Smaller, faster, reasonable Korean performance

**Embedding Models:**
- `intfloat/multilingual-e5-large` - Strong Korean support
- `BAAI/bge-m3` - Multilingual, good for RAG
- `jhgan/ko-sroberta-multitask` - Korean-specific

### 10. Evaluation Metrics Guidance

**Suggested Metrics for RAG:**
- **Retrieval Metrics:**
  - Recall@K (K=1, 3, 5)
  - MRR (Mean Reciprocal Rank)
  - NDCG (Normalized Discounted Cumulative Gain)

- **Generation Metrics:**
  - BLEU, ROUGE (if reference answers available)
  - BERTScore (semantic similarity)
  - Perplexity
  - Human evaluation (qualitative)

- **RAG-Specific:**
  - Faithfulness (answer grounded in retrieved context)
  - Answer relevance
  - Context relevance
  - End-to-end accuracy

**Important:** Document why each metric was chosen and its relevance to the task.

### 11. Fine-tuning Strategy

**Recommended Approaches:**
- **LoRA/QLoRA** - Memory efficient, faster training
- **Full fine-tuning** - Better performance, requires more resources
- **Instruction tuning** - Adapt model to follow RAG instructions

**Training Tips:**
- Start with small learning rates (1e-5 to 5e-5)
- Use gradient accumulation if batch size is limited
- Monitor validation loss to prevent overfitting
- Save checkpoints frequently
- Log training curves for the report

### 12. Hugging Face Integration

**Dataset Upload:**
```python
from datasets import Dataset, DatasetDict

# Create dataset
dataset = Dataset.from_dict({
    "question": [...],
    "context": [...],
    "answer": [...]
})

# Upload to HuggingFace
dataset.push_to_hub("username/jecheon-rag-dataset")
```

**Model Upload:**
```python
model.push_to_hub("username/korean-rag-finetuned")
tokenizer.push_to_hub("username/korean-rag-finetuned")
```

**Important:**
- Include dataset card with description, usage, and license
- Include model card with training details, intended use, and limitations
- Make repositories public for submission

---

## AI Assistant Operational Guidelines

### When Working on This Project

1. **Understand Context:**
   - This is a job application assignment - quality matters
   - Korean language support is critical
   - All deliverables must be complete and well-documented

2. **Prioritize Deliverables:**
   - Focus on meeting all checklist items
   - Ensure reproducibility
   - Document all decisions and rationale

3. **Code Quality:**
   - Write production-quality code, not just prototypes
   - Include error handling and input validation
   - Make code modular and reusable

4. **Experiment Tracking:**
   - Keep detailed logs of all experiments
   - Save all metrics and outputs for comparison
   - Make it easy to reproduce results

5. **Documentation First:**
   - Document as you code, not after
   - Capture design decisions when they're made
   - Take screenshots during evaluation for the report

6. **Resource Management:**
   - Be mindful of RunPod GPU credits
   - Optimize training to avoid waste
   - Use efficient batch sizes and gradient accumulation

7. **Version Control:**
   - Commit frequently with meaningful messages
   - Tag important milestones
   - Push to remote regularly to avoid data loss

8. **Communication:**
   - If unclear about requirements, refer to assignment document
   - For technical questions, contact: dasol@goodganglabs.com
   - Document assumptions made during development

---

## Quick Reference

### Important Files
- **Assignment:** `AI 포지션 과제 안내 2aa86229d43581a688b7c2ed89434dd5.md:1`
- **Source Data:** `data/raw/제천시관광정보책자.pdf`
- **RunPod Guide:** `docs/Runpod설정방법.pdf`

### Important Links
- **Reference Dataset:** https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO
- **Contact Email:** dasol@goodganglabs.com
- **RunPod Credit:** c1w0wzucm2khx7qk0br9

### Git Information
- **Branch:** `claude/claude-md-mhyifajyydv2kwaf-01RitCh541mcZbfHbpV3LGYj`
- **Remote:** `http://local_proxy@127.0.0.1:46328/git/yesinkim/goodganglabs`

### Common Commands

```bash
# Activate environment
source venv/bin/activate

# Run training
python scripts/train.py --config configs/training_config.yaml

# Run evaluation
python scripts/evaluate.py --model path/to/model --data data/test/

# Push to git (with retry on network failure)
git push -u origin claude/claude-md-mhyifajyydv2kwaf-01RitCh541mcZbfHbpV3LGYj
```

---

## Next Steps for Development

1. **Setup Phase:**
   - [ ] Create project directory structure
   - [ ] Set up virtual environment
   - [ ] Create requirements.txt with necessary dependencies
   - [ ] Initialize configuration files

2. **Data Phase:**
   - [ ] Extract text from PDF
   - [ ] Analyze document structure and content
   - [ ] Design question-answer generation strategy
   - [ ] Generate RAG dataset
   - [ ] Split train/test datasets
   - [ ] Upload to Hugging Face

3. **Baseline Phase:**
   - [ ] Select and justify LLM choice
   - [ ] Select and justify embedding model
   - [ ] Define evaluation metrics with rationale
   - [ ] Implement RAG pipeline
   - [ ] Run baseline evaluation
   - [ ] Document results

4. **Fine-tuning Phase:**
   - [ ] Prepare training data format
   - [ ] Configure fine-tuning parameters
   - [ ] Set up training pipeline
   - [ ] Execute fine-tuning
   - [ ] Monitor training progress
   - [ ] Save and upload model

5. **Evaluation Phase:**
   - [ ] Run fine-tuned model evaluation
   - [ ] Compare baseline vs fine-tuned
   - [ ] Generate visualizations
   - [ ] Collect example outputs
   - [ ] Analyze failure cases

6. **Documentation Phase:**
   - [ ] Write comprehensive report
   - [ ] Create visualizations and charts
   - [ ] Take screenshots for comparison
   - [ ] Review all deliverables
   - [ ] Final submission

---

## Notes for Future AI Assistants

This repository represents a well-defined assignment with clear success criteria. When working on this project:

- **Quality over speed:** This is a job application - make it impressive
- **Document everything:** Rationale for decisions is as important as the code
- **Think like a researcher:** Justify choices, compare alternatives, analyze results
- **Korean language matters:** Ensure all Korean text is handled properly
- **Reproducibility is key:** Anyone should be able to run your code and get the same results

The assignment tests both technical skills and ability to communicate complex ML concepts clearly. Balance implementation quality with thorough documentation.

---

**Last Updated:** 2025-11-15
**Version:** 1.1
**Maintained for:** AI Assistants working on GoodGangLabs RAG Assignment
