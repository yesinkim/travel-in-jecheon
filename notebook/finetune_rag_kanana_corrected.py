"""
Finetune-RAG for KANANA1.5-8B (Corrected Version)

논문 방법론 준수:
- arXiv:2505.10792 "Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in RAG"

주요 수정사항:
1. XML 기반 구조화 포맷 사용
2. Unanswerable 질문 추가 (15% 비율)
3. 환각 방지 메커니즘 강화
4. 논문의 프롬프트 구조 준수
"""

import os
import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional

# Transformers & Training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

# Configuration
PROJECT_ROOT = Path("/Users/yesinkim/Bailando/goodganglabs")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "kanana-finetune-rag-corrected"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. DATA PREPARATION (Corrected Format)
# ============================================================================

def load_and_transform_dataset(file_path: Path, is_train: bool = True) -> Dataset:
    """
    Load dataset and transform to Finetune-RAG format

    Key changes from original:
    1. XML-based format
    2. Add unanswerable questions (15% ratio)
    3. Clear oracle vs distractor distinction
    """
    data = []
    unanswerable_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    for i, original_sample in enumerate(lines):
        # Extract oracle (correct) document
        oracle_doc = original_sample.get('correct_doc', {})
        oracle_text = oracle_doc.get('text', '')
        oracle_filename = oracle_doc.get('filename', '제천 관광 정보')

        # Extract distractor documents
        distractor_docs = original_sample.get('distractor_docs', [])

        question = original_sample.get('question', '')
        answer = original_sample.get('answer', '')

        # Strategy 1: Answerable questions (oracle + distractors)
        # Model should learn to identify correct document and answer
        if oracle_text:
            answerable_sample = {
                'question': question,
                'oracle_doc': {
                    'source': oracle_filename,
                    'text': oracle_text
                },
                'distractor_docs': [
                    {
                        'source': d.get('filename', d.get('title', '기타 문서')),
                        'text': d.get('text', '')
                    }
                    for d in distractor_docs[:2]  # Use up to 2 distractors
                ],
                'answer': answer,
                'is_answerable': True
            }
            data.append(answerable_sample)

        # Strategy 2: Unanswerable questions (only distractors, ~15%)
        # Model should learn to refuse when oracle is not provided
        if is_train and distractor_docs and random.random() < 0.15:
            unanswerable_sample = {
                'question': question,
                'oracle_doc': None,  # No oracle document
                'distractor_docs': [
                    {
                        'source': d.get('filename', d.get('title', '기타 문서')),
                        'text': d.get('text', '')
                    }
                    for d in distractor_docs[:2]
                ],
                'answer': '제공된 정보에는 해당 내용이 없습니다.',
                'is_answerable': False
            }
            data.append(unanswerable_sample)
            unanswerable_count += 1

    dataset = Dataset.from_list(data)

    print(f"✓ Dataset loaded: {len(dataset)} examples")
    if is_train:
        answerable = len(dataset) - unanswerable_count
        print(f"  - Answerable: {answerable} ({answerable/len(dataset)*100:.1f}%)")
        print(f"  - Unanswerable: {unanswerable_count} ({unanswerable_count/len(dataset)*100:.1f}%)")

    return dataset


def format_finetune_rag_xml(sample: Dict) -> str:
    """
    Format sample in Finetune-RAG XML structure (Paper-compliant)

    XML Structure (from paper):
    <document>
    <source>...</source>
    <context>...</context>
    </document>

    <question>...</question>

    <answer>...</answer>
    """

    # System instruction (aligned with paper)
    system_prompt = """당신은 제공된 문서(document)를 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

중요한 규칙:
1. 제공된 문서의 내용만을 사용하여 답변하세요
2. 문서에 답변이 없으면 "제공된 정보에는 해당 내용이 없습니다"라고 답변하세요
3. 추측하거나 문서 외부 지식을 사용하지 마세요
4. 답변은 간결하고 정확해야 합니다"""

    # Build documents section
    documents_xml = ""
    all_docs = []

    # Add oracle document (if exists)
    if sample.get('oracle_doc'):
        all_docs.append(sample['oracle_doc'])

    # Add distractor documents
    if sample.get('distractor_docs'):
        all_docs.extend(sample['distractor_docs'])

    # Shuffle to prevent position bias
    random.shuffle(all_docs)

    # Format as XML
    for doc in all_docs:
        doc_xml = f"""<document>
<source>{doc['source']}</source>
<context>
{doc['text']}
</context>
</document>"""
        documents_xml += doc_xml + "\n\n"

    # Complete format
    formatted = f"""### System:
{system_prompt}

### Documents:
{documents_xml.strip()}

### Question:
<question>{sample['question']}</question>

### Answer:
<answer>{sample['answer']}</answer>"""

    return formatted


# ============================================================================
# 2. MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer(model_name: str = "kakaocorp/kanana-1.5-8b-instruct-2505"):
    """Setup model with QLoRA configuration"""

    # QLoRA config (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    return model, tokenizer


# ============================================================================
# 3. TRAINING
# ============================================================================

def train_finetune_rag(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    model,
    tokenizer,
    output_dir: Path
):
    """Train with Finetune-RAG methodology"""

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        warmup_steps=10,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",  # Set to "wandb" if using W&B
    )

    # Data collator (answer-only loss)
    response_template = "### Answer:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        formatting_func=format_finetune_rag_xml,
        data_collator=collator,
        max_seq_length=2048,
        packing=False
    )

    print("="*80)
    print("Starting Finetune-RAG Training")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print(f"  Format: XML-based (paper-compliant)")
    print(f"  Loss: Answer-only")
    print("="*80)

    # Train
    trainer.train()

    # Save
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\n✓ Training complete! Model saved to {output_dir}")

    return trainer


# ============================================================================
# 4. INFERENCE
# ============================================================================

def generate_answer_finetune_rag(
    model,
    tokenizer,
    question: str,
    documents: List[Dict],
    max_new_tokens: int = 256
) -> str:
    """
    Generate answer using Finetune-RAG format

    Args:
        question: User question
        documents: List of dicts with 'source' and 'text' keys
    """

    system_prompt = """당신은 제공된 문서(document)를 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

중요한 규칙:
1. 제공된 문서의 내용만을 사용하여 답변하세요
2. 문서에 답변이 없으면 "제공된 정보에는 해당 내용이 없습니다"라고 답변하세요
3. 추측하거나 문서 외부 지식을 사용하지 마세요
4. 답변은 간결하고 정확해야 합니다"""

    # Format documents as XML
    documents_xml = ""
    for doc in documents:
        doc_xml = f"""<document>
<source>{doc['source']}</source>
<context>
{doc['text']}
</context>
</document>"""
        documents_xml += doc_xml + "\n\n"

    # Create prompt (without answer)
    prompt = f"""### System:
{system_prompt}

### Documents:
{documents_xml.strip()}

### Question:
<question>{question}</question>

### Answer:
<answer>"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer (after <answer> tag)
    if '<answer>' in generated:
        answer = generated.split('<answer>')[-1].strip()
        # Remove closing tag if present
        answer = answer.replace('</answer>', '').strip()
    else:
        answer = generated.split('### Answer:')[-1].strip()

    return answer


# ============================================================================
# 5. EVALUATION
# ============================================================================

def evaluate_hallucination_resistance(
    model,
    tokenizer,
    test_dataset: Dataset,
    num_samples: int = 5
):
    """
    특별 평가: 환각 저항성 테스트

    Unanswerable 질문에 대해 모델이 올바르게 거부하는지 확인
    """
    print("\n" + "="*80)
    print("환각 저항성 평가 (Hallucination Resistance Test)")
    print("="*80)

    # Filter unanswerable questions
    unanswerable = [s for s in test_dataset if not s.get('is_answerable', True)]

    if not unanswerable:
        print("⚠️  Warning: No unanswerable questions in test set")
        return

    print(f"Testing {min(num_samples, len(unanswerable))} unanswerable questions...\n")

    correct_refusals = 0

    for i, sample in enumerate(unanswerable[:num_samples], 1):
        print(f"--- Example {i} ---")
        print(f"Question: {sample['question']}")

        # Prepare documents (only distractors, no oracle)
        documents = [
            {'source': d['source'], 'text': d['text']}
            for d in sample.get('distractor_docs', [])
        ]

        # Generate answer
        answer = generate_answer_finetune_rag(
            model, tokenizer, sample['question'], documents
        )

        print(f"Generated: {answer}")
        print(f"Expected: {sample['answer']}")

        # Check if model correctly refused
        refusal_phrases = [
            '제공된 정보에는',
            '해당 내용이 없습니다',
            '정보가 없습니다',
            '찾을 수 없습니다'
        ]

        is_correct_refusal = any(phrase in answer for phrase in refusal_phrases)

        if is_correct_refusal:
            print("✅ Correct refusal (No hallucination)")
            correct_refusals += 1
        else:
            print("❌ Failed to refuse (Potential hallucination)")

        print()

    refusal_rate = correct_refusals / min(num_samples, len(unanswerable)) * 100
    print("="*80)
    print(f"Refusal Rate: {refusal_rate:.1f}% ({correct_refusals}/{min(num_samples, len(unanswerable))})")
    print(f"Goal: >70% (paper baseline)")
    print("="*80)

    return refusal_rate


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""

    print("="*80)
    print("Finetune-RAG for KANANA1.5-8B (Corrected Implementation)")
    print("Paper: arXiv:2505.10792")
    print("="*80)

    # 1. Load datasets
    print("\n[Step 1] Loading datasets...")
    train_dataset = load_and_transform_dataset(
        DATA_DIR / "chunks" / "train_qa.jsonl",
        is_train=True
    )
    test_dataset = load_and_transform_dataset(
        DATA_DIR / "chunks" / "test_qa.jsonl",
        is_train=False
    )

    # Show sample formatting
    print("\n[Sample Formatting]")
    print("="*80)
    sample_formatted = format_finetune_rag_xml(train_dataset[0])
    print(sample_formatted[:600], "...")
    print("="*80)

    # 2. Setup model
    print("\n[Step 2] Setting up model...")
    model, tokenizer = setup_model_and_tokenizer()

    # 3. Train
    print("\n[Step 3] Training...")
    trainer = train_finetune_rag(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        tokenizer=tokenizer,
        output_dir=MODEL_OUTPUT_DIR
    )

    # 4. Evaluate hallucination resistance
    print("\n[Step 4] Evaluating hallucination resistance...")
    refusal_rate = evaluate_hallucination_resistance(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        num_samples=5
    )

    # 5. Test inference
    print("\n[Step 5] Testing inference...")
    sample = test_dataset[0]

    # Prepare documents
    documents = []
    if sample.get('oracle_doc'):
        documents.append({
            'source': sample['oracle_doc']['source'],
            'text': sample['oracle_doc']['text']
        })
    for d in sample.get('distractor_docs', []):
        documents.append({'source': d['source'], 'text': d['text']})
    random.shuffle(documents)

    answer = generate_answer_finetune_rag(
        model, tokenizer, sample['question'], documents
    )

    print(f"\nQuestion: {sample['question']}")
    print(f"Generated: {answer}")
    print(f"Ground Truth: {sample['answer']}")

    print("\n✓ All steps complete!")
    print(f"✓ Model saved to: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
