# Finetune-RAG for KANANA1.5-8B

이 프로젝트는 논문 **"Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in Retrieval-Augmented Generation"** (arXiv:2505.10792)의 방법론을 KANANA1.5-8B 모델에 적용한 구현입니다.

## 📋 목차

- [개요](#개요)
- [주요 특징](#주요-특징)
- [환경 요구사항](#환경-요구사항)
- [설치](#설치)
- [사용 방법](#사용-방법)
- [프로젝트 구조](#프로젝트-구조)
- [설정 파일](#설정-파일)
- [평가 및 분석](#평가-및-분석)
- [참고 자료](#참고-자료)

## 개요

### Finetune-RAG란?

Finetune-RAG는 RAG(Retrieval-Augmented Generation) 시스템에서 LLM의 환각(hallucination) 문제를 해결하기 위한 fine-tuning 방법론입니다.

**핵심 아이디어:**
1. **XML 기반 구조화 입력**: 문맥(context)과 질문(question)을 명확하게 구분
2. **환각 방지 학습**: "답변할 수 없음" 응답을 학습하여 부정확한 정보 생성 방지
3. **문맥 기반 답변**: 검색된 문맥만을 사용하여 답변 생성

### KANANA1.5-8B란?

Kakao에서 개발한 8B 파라미터 이중언어(한-영) LLM으로, 다음 특징을 가집니다:
- **크기**: 8.03B parameters
- **문맥 길이**: 32K tokens (YaRN으로 128K까지 확장 가능)
- **강점**: 코딩, 수학, 함수 호출, 한국어 처리
- **아키텍처**: Llama 기반
- **라이센스**: Apache 2.0

## 주요 특징

### ✨ 구현된 기능

1. **QLoRA 기반 효율적 학습**
   - 4-bit 양자화로 메모리 사용량 대폭 감소
   - RTX 4090 (24GB)에서 학습 가능
   - LoRA adapter로 빠른 학습 및 배포

2. **XML 기반 데이터 포맷**
   ```xml
   <document>
   <source>제천시 관광정보</source>
   <context>
   의림지는 제천시 송학면에 위치한 역사적인 저수지입니다.
   </context>
   </document>

   <question>의림지는 어디에 있나요?</question>

   <answer>의림지는 제천시 송학면에 위치해 있습니다.</answer>
   ```

3. **환각 방지 메커니즘**
   - Unanswerable 질문 학습 (15% 비율)
   - "제공된 정보에서 답변을 찾을 수 없습니다" 응답 학습
   - 문맥 기반 답변 강제

4. **포괄적인 평가**
   - 답변 정확도 측정
   - 환각 비율 측정
   - Answerable vs Unanswerable 성능 분리 평가

5. **Weights & Biases 통합**
   - 실시간 학습 모니터링
   - 하이퍼파라미터 추적
   - 실험 비교

## 환경 요구사항

### 하드웨어

**권장 환경:**
- **GPU**: RTX 4090 (24GB) 이상
- **대안**: RTX 3090 (24GB), A40 (48GB), A100 (40GB/80GB)
- **RAM**: 32GB 이상
- **저장공간**: 50GB 이상

**QLoRA 사용 시 메모리 요구사항:**
- 모델 로딩: ~5-6GB
- 학습: ~18-20GB
- 여유 공간: ~4GB

### 소프트웨어

- **Python**: 3.11+
- **CUDA**: 11.8+ (GPU 사용 시)
- **운영체제**: Linux (권장), Windows (WSL2), macOS (CPU만)

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/goodganglabs.git
cd goodganglabs
```

### 2. 의존성 설치

**Option A: uv 사용 (권장)**
```bash
uv sync
```

**Option B: pip 사용**
```bash
pip install -r requirements.txt
```

### 3. 설치 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 사용 방법

### 빠른 시작 (Quick Start)

#### 1. 샘플 데이터셋 준비

```bash
bash scripts/prepare_sample_dataset.sh
```

생성 위치: `data/processed/finetune_rag_sample/`

#### 2. 학습 실행

```bash
bash scripts/train_kanana_finetune_rag.sh
```

기본적으로 `configs/finetune_rag_config.yaml` 설정을 사용합니다.

#### 3. 모델 평가

```bash
bash scripts/evaluate_model.sh \
  --model_path models/kanana-finetune-rag \
  --dataset_path data/processed/finetune_rag_sample
```

### 고급 사용법

#### 커스텀 데이터셋으로 학습

1. **데이터셋 준비 스크립트 작성**

```python
from src.data_processing.prepare_finetune_rag_dataset import (
    FinetuneRAGDatasetBuilder,
    RAGExample,
    AnswerType
)
from pathlib import Path

# 예제 생성
examples = [
    RAGExample(
        question="제천의 대표 관광지는?",
        context="제천의 대표 관광지로는 의림지, 청풍호반 케이블카 등이 있습니다.",
        answer="제천의 대표 관광지로는 의림지, 청풍호반 케이블카 등이 있습니다.",
        answer_type=AnswerType.ANSWERABLE,
    ),
    # ... 더 많은 예제 추가
]

# 데이터셋 빌드
builder = FinetuneRAGDatasetBuilder(xml_format=True)
dataset = builder.build_dataset(
    examples,
    output_path=Path("data/processed/my_custom_dataset")
)
```

2. **설정 파일 수정**

`configs/finetune_rag_config.yaml`:
```yaml
dataset:
  train_dataset_path: "data/processed/my_custom_dataset"
```

3. **학습 실행**

```bash
python src/training/finetune_rag.py --config configs/finetune_rag_config.yaml
```

#### PEFT Adapter만 로드하여 평가

```bash
python src/evaluation/evaluate_rag_model.py \
  --model_path models/kanana-finetune-rag \
  --base_model_path kakaocorp/kanana-1.5-8b-base \
  --dataset_path data/processed/finetune_rag_dataset \
  --output_path results/eval_results.json
```

#### 추론 테스트

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "kakaocorp/kanana-1.5-8b-base",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "models/kanana-finetune-rag")
tokenizer = AutoTokenizer.from_pretrained("models/kanana-finetune-rag")

# 추론
question = "의림지는 어디에 있나요?"
context = "의림지는 제천시 송학면에 위치한 역사적인 저수지입니다."

messages = [
    {"role": "system", "content": "제공된 문맥을 바탕으로 질문에 답변하세요."},
    {"role": "user", "content": f"""<document>
<context>{context}</context>
</document>
<question>{question}</question>
<answer>"""}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids, max_new_tokens=256, temperature=0.7)
answer = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

print(f"답변: {answer}")
```

## 프로젝트 구조

```
goodganglabs/
├── configs/
│   └── finetune_rag_config.yaml       # 학습 설정 파일
├── src/
│   ├── data_processing/
│   │   └── prepare_finetune_rag_dataset.py  # 데이터셋 준비
│   ├── training/
│   │   └── finetune_rag.py            # 학습 스크립트
│   └── evaluation/
│       └── evaluate_rag_model.py      # 평가 스크립트
├── scripts/
│   ├── train_kanana_finetune_rag.sh   # 학습 실행 스크립트
│   ├── evaluate_model.sh              # 평가 실행 스크립트
│   └── prepare_sample_dataset.sh      # 샘플 데이터 준비
├── data/
│   ├── raw/                           # 원본 데이터
│   └── processed/                     # 처리된 데이터셋
├── models/                            # 학습된 모델 저장
├── results/                           # 평가 결과
└── docs/
    └── FINETUNE_RAG_README.md         # 이 문서
```

## 설정 파일

### 주요 설정 항목

#### 모델 설정
```yaml
model:
  name: "kakaocorp/kanana-1.5-8b-base"
  torch_dtype: "bfloat16"
  use_flash_attention_2: true
```

#### QLoRA 설정
```yaml
qlora:
  enabled: true
  load_in_4bit: true
  lora_r: 16              # LoRA rank (8, 16, 32, 64)
  lora_alpha: 32          # Scaling factor (보통 2x rank)
  lora_dropout: 0.05
  target_modules:         # 학습할 모듈
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

#### 학습 설정
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch = 16
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  optim: "paged_adamw_32bit"
  bf16: true
```

#### 데이터셋 설정
```yaml
dataset:
  train_dataset_path: "data/processed/finetune_rag_dataset"
  use_chat_template: true
  xml_format: true
  max_seq_length: 2048
```

### 하이퍼파라미터 튜닝 가이드

| 파라미터 | 기본값 | 튜닝 가이드 |
|---------|--------|------------|
| `lora_r` | 16 | 8 (빠름, 저품질) → 64 (느림, 고품질) |
| `learning_rate` | 2e-4 | QLoRA: 1e-4 ~ 5e-4, Full FT: 1e-5 ~ 5e-5 |
| `num_train_epochs` | 3 | 작은 데이터셋: 3-5, 큰 데이터셋: 1-2 |
| `batch_size` | 2 | GPU 메모리에 따라 조정 |
| `max_seq_length` | 2048 | 짧은 문서: 512-1024, 긴 문서: 2048-4096 |

## 평가 및 분석

### 평가 메트릭

1. **Overall Accuracy**: 전체 정확도
2. **Answerable Accuracy**: 답변 가능한 질문의 정확도
3. **Unanswerable Accuracy (Refusal Rate)**: 답변 불가 질문을 올바르게 거부한 비율
4. **Hallucination Rate**: 환각 발생 비율

### 평가 결과 예시

```json
{
  "metrics": {
    "total_examples": 100,
    "accuracy": 0.85,
    "hallucination_rate": 0.08,
    "answerable_examples": 85,
    "answerable_accuracy": 0.88,
    "unanswerable_examples": 15,
    "unanswerable_accuracy": 0.73,
    "refusal_rate": 0.73
  }
}
```

### 베이스라인 vs Fine-tuned 비교

**평가 방법:**
1. 베이스라인 모델로 평가
2. Fine-tuned 모델로 평가
3. 메트릭 비교

```bash
# Baseline 평가
python src/evaluation/evaluate_rag_model.py \
  --model_path kakaocorp/kanana-1.5-8b-base \
  --dataset_path data/processed/finetune_rag_dataset \
  --output_path results/baseline_results.json

# Fine-tuned 평가
python src/evaluation/evaluate_rag_model.py \
  --model_path models/kanana-finetune-rag \
  --base_model_path kakaocorp/kanana-1.5-8b-base \
  --dataset_path data/processed/finetune_rag_dataset \
  --output_path results/finetuned_results.json
```

### 예상 성능 향상

논문 기준 Finetune-RAG 적용 시:
- **Answerable Accuracy**: +5-10%
- **Refusal Rate**: +15-25%
- **Hallucination Rate**: -10-20%

## 비용 추정 (RunPod 기준)

### GPU 가격
- RTX 4090: ~$0.50/hour
- A40: ~$0.60/hour

### 학습 시간 추정

| 데이터셋 크기 | Epochs | 예상 시간 | 예상 비용 (RTX 4090) |
|-------------|--------|---------|-------------------|
| 100 examples | 3 | 1-2 hours | $0.50-1.00 |
| 150 examples | 3 | 2-3 hours | $1.00-1.50 |
| 200 examples | 3 | 3-4 hours | $1.50-2.00 |

**총 예상 비용 (Baseline + Fine-tuning + Evaluation):**
- **최소**: $2-3
- **권장**: $3-5
- **최대**: $5-7

**$15 크레딧으로 충분한 실험 가능!**

## 트러블슈팅

### GPU 메모리 부족 (OOM)

**증상:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**해결 방법:**
1. Batch size 줄이기:
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 16
   ```

2. Sequence length 줄이기:
   ```yaml
   max_seq_length: 1024
   ```

3. LoRA rank 줄이기:
   ```yaml
   lora_r: 8
   ```

### 학습이 너무 느림

**해결 방법:**
1. Flash Attention 2 활성화 (이미 활성화됨)
2. Gradient checkpointing 비활성화 (메모리 충분 시):
   ```yaml
   gradient_checkpointing: false
   ```
3. DataLoader workers 증가:
   ```yaml
   dataloader_num_workers: 8
   ```

### Loss가 수렴하지 않음

**해결 방법:**
1. Learning rate 조정:
   ```yaml
   learning_rate: 1.0e-4  # 더 낮은 값
   ```

2. Warmup 비율 증가:
   ```yaml
   warmup_ratio: 0.1
   ```

3. 데이터 품질 확인:
   - Answerable/Unanswerable 비율 확인
   - 데이터 중복 제거
   - 레이블 정확성 확인

## 참고 자료

### 논문
- **Finetune-RAG**: [arXiv:2505.10792](https://arxiv.org/pdf/2505.10792)
- **KANANA**: [arXiv:2502.18934](https://arxiv.org/abs/2502.18934)
- **QLoRA**: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

### 모델
- [kakaocorp/kanana-1.5-8b-base](https://huggingface.co/kakaocorp/kanana-1.5-8b-base)
- [kakaocorp/kanana-1.5-8b-instruct-2505](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505)

### 라이브러리
- [Transformers](https://huggingface.co/docs/transformers)
- [PEFT](https://huggingface.co/docs/peft)
- [TRL](https://huggingface.co/docs/trl)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

### 기타
- [Allganize Korean RAG Evaluation Dataset](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
- [RunPod Documentation](https://docs.runpod.io/)
- [Weights & Biases](https://wandb.ai/)

## 라이센스

이 프로젝트는 Apache 2.0 라이센스를 따릅니다.

## 문의

- **이메일**: dasol@goodganglabs.com
- **GitHub Issues**: [goodganglabs/issues](https://github.com/your-username/goodganglabs/issues)

---

**마지막 업데이트**: 2025-11-18
**버전**: 1.0
