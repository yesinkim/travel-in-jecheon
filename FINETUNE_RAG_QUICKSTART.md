# Finetune-RAG for KANANA1.5-8B - Quick Start Guide

이 가이드는 KANANA1.5-8B 모델을 Finetune-RAG 방법론으로 빠르게 학습시키는 방법을 설명합니다.

## 🚀 5분 Quick Start

### 1. 의존성 설치

```bash
# UV 사용 (권장)
uv sync

# 또는 pip 사용
pip install -r requirements.txt
```

### 2. 샘플 데이터셋 준비

```bash
bash scripts/prepare_sample_dataset.sh
```

### 3. 학습 실행

```bash
bash scripts/train_kanana_finetune_rag.sh
```

### 4. 모델 평가

```bash
bash scripts/evaluate_model.sh \
  --model_path models/kanana-finetune-rag \
  --dataset_path data/processed/finetune_rag_sample
```

## 📊 예상 결과

### 학습 시간
- **GPU**: RTX 4090
- **데이터**: 5 examples (샘플)
- **예상 시간**: 5-10분
- **메모리 사용량**: ~18GB

### 평가 메트릭 (예상)

```
================================
EVALUATION METRICS
================================
Total Examples: 5
Overall Accuracy: 80.00%
Hallucination Rate: 10.00%

Answerable Questions:
  Count: 4
  Accuracy: 85.00%

Unanswerable Questions:
  Count: 1
  Accuracy (Refusal Rate): 70.00%
================================
```

## 🎯 다음 단계

### 실제 데이터셋 준비

1. **PDF에서 데이터 추출** (제천 관광 정보)
2. **Q&A 쌍 생성** (150-200개 권장)
3. **Unanswerable 질문 추가** (15% 비율)

예제:
```python
from src.data_processing.prepare_finetune_rag_dataset import (
    FinetuneRAGDatasetBuilder,
    RAGExample,
    AnswerType
)

examples = [
    RAGExample(
        question="의림지는 어디에 있나요?",
        context="의림지는 제천시 송학면 의림대로 47길 7에 위치...",
        answer="의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다.",
        answer_type=AnswerType.ANSWERABLE,
    ),
    # ... 더 많은 예제
]

builder = FinetuneRAGDatasetBuilder(xml_format=True)
dataset = builder.build_dataset(
    examples,
    output_path="data/processed/jecheon_rag_dataset"
)
```

### 학습 설정 조정

`configs/finetune_rag_config.yaml` 파일 수정:

```yaml
# 더 많은 데이터로 학습 시
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4

# 데이터셋 경로 변경
dataset:
  train_dataset_path: "data/processed/jecheon_rag_dataset"
```

### HuggingFace Hub에 업로드

학습 완료 후:

```python
from transformers import AutoTokenizer
from peft import PeftModel, AutoModelForCausalLM

# 모델 로드
base_model = AutoModelForCausalLM.from_pretrained("kakaocorp/kanana-1.5-8b-base")
model = PeftModel.from_pretrained(base_model, "models/kanana-finetune-rag")
tokenizer = AutoTokenizer.from_pretrained("models/kanana-finetune-rag")

# Merge adapter (선택사항)
model = model.merge_and_unload()

# HuggingFace Hub에 푸시
model.push_to_hub("your-username/kanana-finetune-rag-jecheon")
tokenizer.push_to_hub("your-username/kanana-finetune-rag-jecheon")
```

## 📝 중요 사항

### GPU 메모리 관리

**메모리 부족 시:**
```yaml
# configs/finetune_rag_config.yaml
training:
  per_device_train_batch_size: 1  # 2 → 1로 감소
  gradient_accumulation_steps: 16  # 8 → 16으로 증가
```

### Wandb 설정 (선택사항)

학습 과정을 시각화하려면:

```bash
# Wandb 로그인
wandb login

# configs/finetune_rag_config.yaml에서 활성화
wandb:
  enabled: true
  project: "kanana-finetune-rag"
  name: "experiment-1"
```

## 🔍 트러블슈팅

### CUDA Out of Memory
→ Batch size 줄이기, Sequence length 줄이기 (2048 → 1024)

### 학습이 너무 느림
→ Flash Attention 2 확인, DataLoader workers 증가

### Loss가 수렴하지 않음
→ Learning rate 조정 (2e-4 → 1e-4), Warmup 비율 증가

## 📚 더 자세한 정보

- **전체 문서**: [docs/FINETUNE_RAG_README.md](docs/FINETUNE_RAG_README.md)
- **논문**: [arXiv:2505.10792](https://arxiv.org/pdf/2505.10792)
- **KANANA 모델**: [HuggingFace](https://huggingface.co/kakaocorp/kanana-1.5-8b-base)

---

**Happy Fine-tuning! 🎉**
