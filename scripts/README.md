# Jecheon Tourism RAG Dataset Generation Pipeline

제천시 관광 정보를 바탕으로 RAG(Retrieval-Augmented Generation) 학습용 데이터셋을 생성하는 파이프라인입니다.

## 📋 개요

이 파이프라인은 **Finetune-RAG** 방법론을 따라 다음과 같은 데이터셋을 생성합니다:

- ✅ **문서 Chunks**: 제천 관광 정보를 의미 단위로 분할
- ✅ **Q&A Pairs**: Claude API를 사용한 고품질 질문-답변 생성
- ✅ **Distractor Documents**: 학습 효과를 높이기 위한 오답 문서 추가
- ✅ **Training Format**: Instruction tuning 및 Hugging Face 업로드 형식

## 🎯 생성 목표

- **총 문서 개수**: 31개 (RAG 최적화 청킹)
- **청크 크기**: 300-2000 chars (맥락 보존 + 노이즈 최소화)
- **총 Q&A 개수**: 120-150개
- **질문 타입 분포**:
  - Factual (사실 질문): 40%
  - Descriptive (설명 질문): 30%
  - Recommendation (추천 질문): 15%
  - Comparison (비교 질문): 10%
  - No-answer (정보 없음): 5%

- **Distractor 개수**: 2개 (Finetune-RAG 표준)
  - Hard distractor: 1개 (같은 카테고리)
  - Easy distractor: 1개 (다른 카테고리)
  - 총 문서/Q&A: 3개 (1 correct + 2 distractors)

- **Train/Test Split**: 79% / 21% (stratified)

## 📁 파일 구조

```
scripts/
├── README.md                        # 이 파일
├── run_pipeline.py                  # 전체 파이프라인 실행 스크립트
├── 01_extract_pdf_chunks.py         # Step 1: 문서 청킹
├── 02_generate_qa_with_claude.py    # Step 2: Q&A 생성 (Claude API)
├── 03_add_distractors.py            # Step 3: Distractor 추가
├── 04_format_training_data.py       # Step 4: 학습 데이터 포맷팅
└── 05_split_train_test.py           # Step 5: Train/Test 분할

data/
├── chunks/
│   ├── documents.jsonl              # 문서 chunks
│   ├── qa_pairs.jsonl               # Q&A pairs (Claude 생성)
│   └── qa_with_distractors.jsonl    # Q&A + distractors
└── processed/
    ├── training_data.jsonl          # Instruction tuning 형식
    ├── dataset_hf.jsonl             # Hugging Face 업로드 형식
    ├── train.jsonl                  # 학습 데이터
    └── test.jsonl                   # 평가 데이터
```

## 🚀 사용 방법

### 1. 사전 준비

#### API 키 설정 (필수)

Claude API를 사용하여 Q&A를 생성하므로 Anthropic API 키가 필요합니다:

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

#### Dependencies 확인

이미 `uv add anthropic tqdm`로 설치되어 있어야 합니다.

### 2. 전체 파이프라인 실행

```bash
# 전체 파이프라인 실행
python scripts/run_pipeline.py
```

### 3. 단계별 실행 (선택사항)

각 스크립트를 개별적으로 실행할 수도 있습니다:

```bash
# Step 1: 문서 청킹
python scripts/01_extract_pdf_chunks.py

# Step 2: Q&A 생성 (Claude API 필요)
python scripts/02_generate_qa_with_claude.py

# Step 3: Distractor 추가
python scripts/03_add_distractors.py

# Step 4: 학습 데이터 포맷팅
python scripts/04_format_training_data.py

# Step 5: Train/Test 분할
python scripts/05_split_train_test.py
```

### 4. Q&A 생성 단계 스킵 (이미 생성된 경우)

```bash
# Q&A 생성을 제외하고 나머지 단계만 실행
python scripts/run_pipeline.py --skip-qa-generation
```

## 📊 출력 데이터 형식

### 1. documents.jsonl

```json
{
  "doc_id": "doc_001",
  "title": "의림지·의림지역사박물관",
  "category": "tourism",
  "content": "의림지는 제천 10경 중 하나로...",
  "metadata": {
    "page": 12,
    "location": "송학면",
    "address": "제천시 송학면 의림대로 47길 7"
  },
  "filename": "doc_001_의림지·의림지역사박물관.txt"
}
```

### 2. qa_pairs.jsonl

```json
{
  "question": "의림지는 어디에 있나요?",
  "answer": "의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다.",
  "question_type": "factual",
  "difficulty": "easy",
  "doc_id": "doc_001",
  "doc_title": "의림지·의림지역사박물관",
  "doc_category": "tourism",
  "doc_content": "..."
}
```

### 3. qa_with_distractors.jsonl

```json
{
  "question": "의림지는 어디에 있나요?",
  "answer": "의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다.",
  "question_type": "factual",
  "difficulty": "easy",
  "correct_doc_id": "doc_001",
  "correct_doc": { "doc_id": "doc_001", ... },
  "distractor_docs": [
    { "doc_id": "doc_002", ... },
    { "doc_id": "doc_015", ... },
    { "doc_id": "doc_008", ... }
  ]
}
```

### 4. training_data.jsonl (Instruction Tuning)

```json
{
  "instruction": "제천 관광 정보를 바탕으로 질문에 답하세요...",
  "documents": "<Documents>\n  <Document id=\"doc_001\">...",
  "question": "의림지는 어디에 있나요?",
  "answer": "의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다.",
  "full_prompt": "...",
  "question_type": "factual",
  "difficulty": "easy",
  "correct_doc_id": "doc_001"
}
```

### 5. dataset_hf.jsonl (Hugging Face Format)

Finetune-RAG 데이터셋 구조를 따릅니다:

```json
{
  "question": "의림지는 어디에 있나요?",
  "answer": "의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다.",
  "content": "정답 문서 내용...",
  "filename": "doc_001_의림지·의림지역사박물관.txt",
  "fictitious_content1": "오답 문서 1 내용...",
  "fictitious_filename1": "doc_002_청풍호반케이블카.txt",
  "fictitious_content2": "오답 문서 2 내용...",
  "fictitious_filename2": "doc_015_제천맛집소개.txt",
  "question_type": "factual",
  "difficulty": "easy"
}
```

## 🔧 커스터마이징

### Q&A 생성 개수 조정

`scripts/02_generate_qa_with_claude.py`의 `QUESTIONS_PER_CHUNK` 수정:

```python
QUESTIONS_PER_CHUNK = {
    "tourism": 8,  # 관광지당 8개 질문
    "transportation": 7,
    "food": 8,
    # ...
}
```

### Distractor 개수 조정

`scripts/03_add_distractors.py`의 초기화 파라미터 수정:

```python
adder = DistractorAdder(
    num_distractors=3,  # distractor 개수
    hard_ratio=0.3      # 같은 카테고리 distractor 비율
)
```

### Train/Test Split 비율 조정

`scripts/05_split_train_test.py`의 초기화 파라미터 수정:

```python
splitter = TrainTestSplitter(
    test_size=0.21,  # 21% test set
    random_seed=42   # 재현성을 위한 seed
)
```

### 문서 포맷 변경

`scripts/04_format_training_data.py`에서 포맷 스타일 선택:

```python
# XML 스타일 (기본)
instruction_data, hf_data = formatter.format_all_data(format_style="xml")

# Baseline 스타일
instruction_data, hf_data = formatter.format_all_data(format_style="baseline")
```

## 📈 예상 소요 시간 및 비용

| 단계 | 소요 시간 | 비용 (Claude API) |
|-----|----------|-----------------|
| 문서 청킹 | ~10초 | 무료 |
| Q&A 생성 | ~5-8분 | $2-4 |
| Distractor 추가 | ~5초 | 무료 |
| 데이터 포맷팅 | ~5초 | 무료 |
| Train/Test 분할 | ~2초 | 무료 |
| **전체** | **~6-10분** | **$2-4** |

- 31개 documents × 평균 4-5 Q&A = ~124-155 Q&A pairs
- Claude API 요청: 31회 (rate limiting 적용)
- 예상 토큰 사용량: ~80K input + ~150K output = ~$2-4
- Distractor: 2개/Q&A (Finetune-RAG 표준)

## 🎨 데이터 품질 관리

### Distractor 선택 전략 (Finetune-RAG 표준)

**총 3개 문서**: 1 correct + 2 distractors

- **Hard distractor (50%, 1개)**: 같은 카테고리 내 다른 문서
  - 예: "의림지" 질문 → "청풍호반 케이블카" distractor (둘 다 tourism)
  - 목적: 모델이 미세한 차이를 구분하도록 학습

- **Easy distractor (50%, 1개)**: 완전히 다른 카테고리 문서
  - 예: "의림지" 질문 → "제천맛집" distractor (tourism vs food)
  - 목적: 기본적인 주제 구분 능력 학습

### Stratified Sampling

Train/Test 분할 시 질문 타입 분포를 유지합니다:
- Factual: Train 40% → Test 40%
- Descriptive: Train 30% → Test 30%
- 등등...

### 재현성 보장

- Random seed 고정: `random_seed=42`
- 같은 seed로 실행하면 항상 같은 train/test split

## 🐛 문제 해결

### API 키 오류

```
❌ Error: ANTHROPIC_API_KEY environment variable not set!
```

**해결**: API 키를 환경변수로 설정
```bash
export ANTHROPIC_API_KEY='your-api-key'
```

### Rate Limiting

Claude API는 rate limit이 있습니다. 스크립트에 1초 delay가 포함되어 있습니다.

**429 에러 발생 시**:
- `scripts/02_generate_qa_with_claude.py`의 `time.sleep(1)`을 `time.sleep(2)`로 증가

### JSON Parsing 오류

Claude가 잘못된 JSON을 반환하는 경우가 있습니다.

**해결**: 스크립트가 자동으로 재시도하지만, 실패 시 해당 문서를 건너뜁니다.

## 📚 참고 자료

- **Finetune-RAG 논문**: [링크 필요]
- **Finetune-RAG 데이터셋**: https://huggingface.co/datasets/pints-ai/Finetune-RAG
- **Anthropic Claude API**: https://docs.anthropic.com/

## 🤝 기여

버그 리포트나 개선 제안은 이슈로 등록해주세요!

## 📄 라이센스

MIT License
