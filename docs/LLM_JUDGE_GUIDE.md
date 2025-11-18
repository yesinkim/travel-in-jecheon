# LLM-as-a-Judge 평가 가이드 (Bench-RAG)

이 문서는 Bench-RAG 스타일의 LLM-as-a-Judge 평가 시스템 사용법을 설명합니다.

## 📊 평가 메트릭 (4가지)

### 1. Accuracy (정확성)
- **타입:** Boolean (true/false)
- **측정:** 문서에 없는 내용을 추가했는지 (환각 여부)
- **판정:**
  - `true` = 문서 내용만 사용, 추가 정보 없음
  - `false` = 문서에 없는 내용을 추가함 (환각 발생)

**예시:**
```
문서: "의림지는 제천시 송학면에 위치합니다."
질문: "의림지는 어디에 있나요?"

✓ 좋은 답변: "제천시 송학면에 위치합니다." → accuracy: true
✗ 나쁜 답변: "제천시 송학면에 위치하며, 입장료는 무료입니다." → accuracy: false
  (입장료 정보는 문서에 없음)
```

### 2. Helpfulness (유용성)
- **타입:** Integer (1-10)
- **측정:** 답변이 얼마나 도움이 되는지
- **점수:**
  - 1-3: 도움 안 됨
  - 4-6: 보통
  - 7-8: 도움 됨
  - 9-10: 매우 도움 됨

### 3. Relevance (관련성)
- **타입:** Integer (1-10)
- **측정:** 질문에 얼마나 적합한 답변인지
- **점수:**
  - 1-3: 질문과 관련 없음
  - 4-6: 부분적으로 관련
  - 7-8: 관련성 높음
  - 9-10: 완벽하게 답변

### 4. Depth (깊이)
- **타입:** Integer (1-10)
- **측정:** 답변의 상세도와 깊이
- **점수:**
  - 1-3: 너무 짧거나 피상적
  - 4-6: 기본적인 정보 포함
  - 7-8: 상세한 정보
  - 9-10: 매우 상세하고 포괄적

---

## 🚀 사용 방법

### 전체 워크플로우

```
모델 추론 결과
    ↓
[1] prepare_predictions_for_judge.py (형식 변환)
    ↓
Judge 형식 데이터
    ↓
[2] evaluate_with_llm_judge.py (LLM 평가)
    ↓
평가 결과 (Accuracy, Helpfulness, Relevance, Depth)
```

### Step 1: 모델 예측 결과 준비

먼저 모델의 추론 결과가 필요합니다:

**필요한 파일:**
- `test_data.jsonl`: 테스트 데이터셋 (질문, 정답 문서, 정답)
- `model_predictions.jsonl`: 모델이 생성한 답변

**test_data.jsonl 형식:**
```json
{
  "question": "의림지는 어디에 있나요?",
  "answer": "제천시 송학면 의림대로 47길 7에 위치합니다.",
  "documents": [
    {
      "doc_id": "doc_001",
      "content": "의림지는 제천시 송학면 의림대로 47길 7에 위치...",
      "is_correct": true
    },
    {
      "doc_id": "doc_002",
      "content": "청풍호는 충주댐 건설로...",
      "is_correct": false
    }
  ]
}
```

**model_predictions.jsonl 형식:**
```json
{
  "question": "의림지는 어디에 있나요?",
  "generated_answer": "의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다."
}
```

### Step 2: Judge 형식으로 변환

```bash
python scripts/prepare_predictions_for_judge.py \
  --dataset data/processed/test_data.jsonl \
  --predictions outputs/baseline_predictions.jsonl \
  --output outputs/baseline_for_judge.jsonl
```

**출력 형식 (for_judge.jsonl):**
```json
{
  "filename": "제천시관광정보책자.pdf",
  "content": "의림지는 제천시 송학면 의림대로 47길 7에 위치...",
  "question": "의림지는 어디에 있나요?",
  "response": "의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다."
}
```

### Step 3: LLM Judge로 평가

```bash
python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/baseline_for_judge.jsonl \
  --output outputs/baseline_judge_results.jsonl \
  --model gemini-2.0-flash-exp
```

**옵션:**
- `--predictions`: Judge 형식 예측 파일 경로
- `--output`: 결과 저장 경로
- `--model`: 사용할 Gemini 모델 (기본: gemini-2.0-flash-exp)
- `--limit`: 테스트용으로 일부만 평가 (예: --limit 10)

**평가 시간:**
- 샘플 1개당 **4번의 API 호출** (Accuracy, Helpfulness, Relevance, Depth)
- 30개 샘플 = 120번 API 호출 ≈ **2-3분 소요**

### Step 4: 결과 확인

**콘솔 출력:**
```
============================================================
LLM-AS-A-JUDGE EVALUATION RESULTS (Bench-RAG)
============================================================

📊 Aggregate Metrics:
  Accuracy Rate:     85.00%
  Avg Helpfulness:   8.50/10
  Avg Relevance:     9.20/10
  Avg Depth:         7.80/10

📈 Evaluation Stats:
  Successfully Evaluated: 30
  Failed Evaluations:     0

============================================================
```

**결과 파일 (baseline_judge_results.jsonl):**
```json
{
  "filename": "제천시관광정보책자.pdf",
  "question": "의림지는 어디에 있나요?",
  "response": "의림지는 제천시 송학면 의림대로 47길 7에 위치해 있습니다.",
  "accuracy": true,
  "accuracy_explanation": "응답은 제공된 정보에만 기반하여 작성되었으며, 추가 세부 정보를 포함하지 않습니다.",
  "helpfulness": 9,
  "helpfulness_explanation": "응답은 질문에 대한 정확한 주소를 제공하여 매우 유용합니다.",
  "relevance": 10,
  "relevance_explanation": "응답은 질문에 완벽하게 답변합니다.",
  "depth": 7,
  "depth_explanation": "응답은 간결하지만 필요한 정보를 제공합니다."
}
```

**통계 파일 (baseline_judge_results_aggregate.json):**
```json
{
  "accuracy_rate": 0.85,
  "avg_helpfulness": 8.5,
  "avg_relevance": 9.2,
  "avg_depth": 7.8,
  "num_evaluated": 30,
  "num_failed": 0
}
```

---

## 📈 Baseline vs Fine-tuned 비교

### 1. Baseline 모델 평가

```bash
# Step 1: 형식 변환
python scripts/prepare_predictions_for_judge.py \
  --dataset data/processed/test_data.jsonl \
  --predictions outputs/baseline_predictions.jsonl \
  --output outputs/baseline_for_judge.jsonl

# Step 2: 평가
python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/baseline_for_judge.jsonl \
  --output outputs/baseline_judge_results.jsonl
```

### 2. Fine-tuned 모델 평가

```bash
# Step 1: 형식 변환
python scripts/prepare_predictions_for_judge.py \
  --dataset data/processed/test_data.jsonl \
  --predictions outputs/finetuned_predictions.jsonl \
  --output outputs/finetuned_for_judge.jsonl

# Step 2: 평가
python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/finetuned_for_judge.jsonl \
  --output outputs/finetuned_judge_results.jsonl
```

### 3. 결과 비교

**비교 스크립트 (간단한 Python):**
```python
import json

# Load aggregate results
with open('outputs/baseline_judge_results_aggregate.json') as f:
    baseline = json.load(f)

with open('outputs/finetuned_judge_results_aggregate.json') as f:
    finetuned = json.load(f)

# Compare
print("Metric           | Baseline | Fine-tuned | Improvement")
print("-----------------|----------|------------|------------")
print(f"Accuracy Rate    | {baseline['accuracy_rate']:.2%}   | {finetuned['accuracy_rate']:.2%}     | +{(finetuned['accuracy_rate']-baseline['accuracy_rate'])*100:.1f}pp")
print(f"Helpfulness      | {baseline['avg_helpfulness']:.2f}/10  | {finetuned['avg_helpfulness']:.2f}/10    | +{finetuned['avg_helpfulness']-baseline['avg_helpfulness']:.2f}")
print(f"Relevance        | {baseline['avg_relevance']:.2f}/10  | {finetuned['avg_relevance']:.2f}/10    | +{finetuned['avg_relevance']-baseline['avg_relevance']:.2f}")
print(f"Depth            | {baseline['avg_depth']:.2f}/10  | {finetuned['avg_depth']:.2f}/10    | +{finetuned['avg_depth']-baseline['avg_depth']:.2f}")
```

**예상 출력:**
```
Metric           | Baseline | Fine-tuned | Improvement
-----------------|----------|------------|------------
Accuracy Rate    | 75.00%   | 90.00%     | +15.0pp
Helpfulness      | 7.20/10  | 8.80/10    | +1.60
Relevance        | 8.00/10  | 9.30/10    | +1.30
Depth            | 6.50/10  | 8.20/10    | +1.70
```

---

## 💡 Tips & Best Practices

### 1. 샘플링 전략

**테스트용 (빠른 확인):**
```bash
# 처음 10개만 평가
python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/for_judge.jsonl \
  --output outputs/test_results.jsonl \
  --limit 10
```

**최종 평가 (전체):**
```bash
# 전체 테스트셋 평가
python scripts/evaluate_with_llm_judge.py \
  --predictions outputs/for_judge.jsonl \
  --output outputs/final_results.jsonl
```

### 2. API 비용 절감

**추천 모델 순서 (빠름 → 느림, 저렴 → 비쌈):**
1. `gemini-2.0-flash-exp` (기본, 빠르고 저렴)
2. `gemini-1.5-flash` (안정적)
3. `gemini-1.5-pro` (고품질, 비쌈)

**예상 비용 (30개 샘플):**
- gemini-2.0-flash: **무료** (현재 실험 모델)
- gemini-1.5-flash: ~$0.01-0.02
- gemini-1.5-pro: ~$0.05-0.10

### 3. 재시도 로직

LLM Judge는 자동 재시도 기능이 내장되어 있습니다:
- 실패 시 3번까지 재시도
- 지수 백오프 (2초, 4초, 8초)
- 3번 실패 후 해당 샘플은 결과에 `null` 값으로 저장

### 4. 결과 분석

**정성적 분석:**
```python
import json

# Load detailed results
with open('outputs/judge_results.jsonl') as f:
    results = [json.loads(line) for line in f]

# Find low accuracy cases
low_accuracy = [r for r in results if not r['accuracy']]

print(f"Found {len(low_accuracy)} hallucination cases:")
for case in low_accuracy[:5]:
    print(f"\nQuestion: {case['question']}")
    print(f"Response: {case['response']}")
    print(f"Explanation: {case['accuracy_explanation']}")
```

---

## 🔧 Troubleshooting

### 문제 1: API Key 에러
```
ValueError: GOOGLE_API_KEY environment variable not set
```

**해결:**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

또는 `.env` 파일에 추가:
```
GOOGLE_API_KEY=your-api-key-here
```

### 문제 2: JSON 파싱 에러
```
json.decoder.JSONDecodeError: Expecting value
```

**원인:** LLM이 JSON이 아닌 텍스트를 반환
**해결:**
- 더 안정적인 모델 사용 (gemini-1.5-flash)
- 재시도 로직이 자동으로 처리

### 문제 3: Rate Limit 에러
```
429 Resource Exhausted
```

**해결:**
- `llm_judge.py`의 `time.sleep(0.5)` 값을 늘리기 (예: 1.0)
- 또는 `--limit` 옵션으로 배치 크기 줄이기

---

## 📊 리포트 작성 예시

### Bench-RAG 평가 결과 (리포트 포함 내용)

**표 1: 정량적 평가 결과**

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Accuracy Rate | 75.0% | 90.0% | +15.0pp |
| Helpfulness | 7.2/10 | 8.8/10 | +1.6 |
| Relevance | 8.0/10 | 9.3/10 | +1.3 |
| Depth | 6.5/10 | 8.2/10 | +1.7 |

**표 2: 정성적 분석 (3가지 예시)**

**예시 1: 환각 감소**
- **질문:** "의림지 입장료는 얼마인가요?"
- **Baseline:** "의림지 입장료는 무료입니다." ❌ (Accuracy: false)
- **Fine-tuned:** "제공된 정보에는 입장료에 대한 내용이 없습니다." ✓ (Accuracy: true)

**예시 2: 상세도 개선**
- **질문:** "청풍호반 케이블카는 얼마나 길어요?"
- **Baseline:** "청풍호반 케이블카는 2.3km입니다." (Depth: 5/10)
- **Fine-tuned:** "청풍호반 케이블카는 왕복 2.3km의 거리를 운행하며, 청풍호의 아름다운 경치를 감상할 수 있습니다." (Depth: 8/10)

**예시 3: 관련성 향상**
- **질문:** "제천에서 가을에 가기 좋은 곳은?"
- **Baseline:** "제천에는 의림지, 청풍호 등 많은 관광지가 있습니다." (Relevance: 6/10)
- **Fine-tuned:** "가을에는 백운권 힐링 코스를 추천합니다. 단풍이 아름다운 월악산과 청풍호반을 둘러볼 수 있습니다." (Relevance: 9/10)

---

## 📝 Summary

**LLM-as-a-Judge 평가는:**
- ✅ 자동 메트릭(ROUGE, BERTScore)보다 **인간 평가에 가까움**
- ✅ **환각(hallucination) 탐지**에 특히 효과적
- ✅ **상세한 설명**을 제공하여 정성적 분석 가능
- ⚠️ API 호출 비용 발생 (하지만 gemini-flash는 저렴)
- ⚠️ 평가 시간이 길음 (30개 = 2-3분)

**추천 사용 시나리오:**
1. **빠른 자동 평가:** ROUGE + BERTScore (metrics.py)
2. **심층 품질 평가:** LLM-as-a-Judge (llm_judge.py)
3. **리포트 작성:** 둘 다 사용 + 정성적 예시 분석

---

**작성일:** 2025-11-18
**버전:** 1.0
**연락처:** 과제 관련 문의 - dasol@goodganglabs.com
