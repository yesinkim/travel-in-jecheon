# LLM-as-a-Judge 빠른 시작 가이드

## 🎯 3단계로 RAG 평가하기

### 1️⃣ API Key 설정 (한 번만)

```bash
# .env 파일에 추가
echo "GOOGLE_API_KEY=your-key-here" >> .env
```

### 2️⃣ 모델 예측 결과 준비

**필요한 것:**
- 테스트 데이터셋 (질문, 정답 문서, 정답 포함)
- 모델 추론 결과 (생성된 답변)

```bash
# 예측 결과를 Judge 형식으로 변환
python scripts/prepare_predictions_for_judge.py \
  --dataset data/processed/test_data.jsonl \
  --predictions outputs/model_predictions.jsonl \
  --output outputs/for_judge.jsonl
```

### 3️⃣ 평가 실행

```bash
# LLM Judge로 평가
python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/for_judge.jsonl \
  --output outputs/judge_results.jsonl
```

**출력:**
```
============================================================
LLM-AS-A-JUDGE EVALUATION RESULTS (Bench-RAG)
============================================================

📊 Aggregate Metrics:
  Accuracy Rate:     85.00%    ← 환각 없는 답변 비율
  Avg Helpfulness:   8.50/10   ← 얼마나 도움되는지
  Avg Relevance:     9.20/10   ← 질문과 관련성
  Avg Depth:         7.80/10   ← 답변 상세도

📈 Evaluation Stats:
  Successfully Evaluated: 30
  Failed Evaluations:     0
============================================================
```

---

## 📊 Baseline vs Fine-tuned 비교

```bash
# 1. Baseline 평가
python scripts/prepare_predictions_for_judge.py \
  --dataset data/processed/test_data.jsonl \
  --predictions outputs/baseline_predictions.jsonl \
  --output outputs/baseline_for_judge.jsonl

python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/baseline_for_judge.jsonl \
  --output outputs/baseline_results.jsonl

# 2. Fine-tuned 평가
python scripts/prepare_predictions_for_judge.py \
  --dataset data/processed/test_data.jsonl \
  --predictions outputs/finetuned_predictions.jsonl \
  --output outputs/finetuned_for_judge.jsonl

python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/finetuned_for_judge.jsonl \
  --output outputs/finetuned_results.jsonl
```

**결과 비교:**
```python
import json

# Load results
with open('outputs/baseline_results_aggregate.json') as f:
    baseline = json.load(f)
with open('outputs/finetuned_results_aggregate.json') as f:
    finetuned = json.load(f)

# Print comparison
print(f"Accuracy:    {baseline['accuracy_rate']:.1%} → {finetuned['accuracy_rate']:.1%}")
print(f"Helpfulness: {baseline['avg_helpfulness']:.1f} → {finetuned['avg_helpfulness']:.1f}")
print(f"Relevance:   {baseline['avg_relevance']:.1f} → {finetuned['avg_relevance']:.1f}")
print(f"Depth:       {baseline['avg_depth']:.1f} → {finetuned['avg_depth']:.1f}")
```

---

## 💡 주요 포인트

### ✅ 장점
- **환각 탐지**: 문서에 없는 내용 추가했는지 자동으로 확인
- **인간 평가에 가까움**: ROUGE보다 실제 품질을 잘 반영
- **상세한 설명**: 왜 그런 점수를 받았는지 설명 제공

### ⚠️ 주의사항
- **비용**: 샘플 1개당 4번 API 호출 (하지만 gemini-flash는 거의 무료)
- **시간**: 30개 평가 = 약 2-3분 소요
- **API Key 필요**: `.env`에 `GOOGLE_API_KEY` 설정

### 🎯 사용 시나리오

**개발 중 (빠른 확인):**
```bash
# 처음 5개만 테스트
python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/for_judge.jsonl \
  --output outputs/test.jsonl \
  --limit 5
```

**최종 평가 (리포트용):**
```bash
# 전체 테스트셋 평가
python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/for_judge.jsonl \
  --output outputs/final_results.jsonl
```

---

## 📄 데이터 형식

### 입력 (for_judge.jsonl)
```json
{
  "filename": "제천시관광정보책자.pdf",
  "content": "의림지는 제천시 송학면 의림대로 47길 7에 위치...",
  "question": "의림지는 어디에 있나요?",
  "response": "제천시 송학면 의림대로 47길 7에 위치합니다."
}
```

### 출력 (judge_results.jsonl)
```json
{
  "filename": "제천시관광정보책자.pdf",
  "question": "의림지는 어디에 있나요?",
  "response": "제천시 송학면 의림대로 47길 7에 위치합니다.",
  "accuracy": true,
  "accuracy_explanation": "문서 내용만 사용하여 답변함",
  "helpfulness": 9,
  "helpfulness_explanation": "정확한 주소를 제공하여 매우 도움됨",
  "relevance": 10,
  "relevance_explanation": "질문에 완벽하게 답변함",
  "depth": 7,
  "depth_explanation": "필요한 정보를 간결하게 제공"
}
```

---

## 🚨 Troubleshooting

**문제: API Key 에러**
```
ValueError: GOOGLE_API_KEY environment variable not set
```
→ `.env` 파일에 `GOOGLE_API_KEY=your-key` 추가

**문제: Rate Limit**
```
429 Resource Exhausted
```
→ `--limit 10` 옵션으로 배치 크기 줄이기

**문제: JSON 파싱 에러**
→ 더 안정적인 모델 사용: `--model gemini-1.5-flash`

---

**상세 가이드:** `docs/LLM_JUDGE_GUIDE.md` 참고
**작성일:** 2025-11-18
