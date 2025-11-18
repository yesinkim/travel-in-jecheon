# RAG 시스템 평가 가이드

## Overview

RAG 시스템은 **Retrieval (검색)**과 **Generation (생성)** 두 단계로 구성되므로, 각각을 평가해야 합니다.

## 📊 RAG 평가 구조

```
                RAG 시스템
                    │
        ┌───────────┴───────────┐
        │                       │
    Retrieval              Generation
    (검색 평가)             (생성 평가)
        │                       │
    ┌───┴───┐              ┌───┴───┐
    │       │              │       │
  정량적  정성적          정량적  정성적
```

---

## 1️⃣ Retrieval 평가 (검색 품질)

### A. 정량적 메트릭

#### 1. **Recall@K** (재현율)
> 관련 문서 중 실제로 검색된 비율

```python
# 예시: 관련 문서 5개 중 3개 검색
# Recall@3 = 3/5 = 0.6

def recall_at_k(retrieved_docs, relevant_docs, k):
    """
    Args:
        retrieved_docs: 검색된 상위 k개 문서 ID
        relevant_docs: 실제 관련 있는 문서 ID 집합
    """
    retrieved_k = set(retrieved_docs[:k])
    relevant = set(relevant_docs)

    return len(retrieved_k & relevant) / len(relevant)
```

**사용 시기:**
- 검색 시스템의 포괄성 평가
- "관련 정보를 얼마나 많이 찾았는가?"

#### 2. **Precision@K** (정밀도)
> 검색된 문서 중 실제 관련 있는 문서 비율

```python
# 예시: 3개 검색 중 2개가 관련 있음
# Precision@3 = 2/3 = 0.67

def precision_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = set(retrieved_docs[:k])
    relevant = set(relevant_docs)

    return len(retrieved_k & relevant) / k
```

**사용 시기:**
- 검색 정확도 평가
- "검색 결과가 얼마나 정확한가?"

#### 3. **MRR (Mean Reciprocal Rank)** ⭐ 추천
> 첫 번째 관련 문서의 순위 역수 평균

```python
# 예시: 첫 관련 문서가 2번째 위치
# RR = 1/2 = 0.5

def mean_reciprocal_rank(queries_results):
    """
    Args:
        queries_results: [
            ([retrieved_doc_ids], [relevant_doc_ids]),
            ...
        ]
    """
    reciprocal_ranks = []

    for retrieved, relevant in queries_results:
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

**사용 시기:**
- 가장 관련 있는 문서가 상위에 있는지 평가
- RAG에서 가장 중요! (top-1이 답변 품질에 큰 영향)

#### 4. **NDCG (Normalized Discounted Cumulative Gain)**
> 순위와 관련성을 모두 고려한 평가

```python
import numpy as np

def dcg_at_k(relevances, k):
    """
    Args:
        relevances: 검색된 문서들의 관련성 점수 리스트 [3, 2, 3, 0, 1, ...]
                   (0: 무관, 1: 약간 관련, 2: 관련, 3: 매우 관련)
    """
    relevances = np.array(relevances)[:k]
    if relevances.size == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i+1))
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)

def ndcg_at_k(retrieved_relevances, ideal_relevances, k):
    """
    Args:
        retrieved_relevances: 실제 검색 순서대로 관련성 점수
        ideal_relevances: 이상적인 순서 (관련성 높은 순)
    """
    dcg = dcg_at_k(retrieved_relevances, k)
    idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)

    return dcg / idcg if idcg > 0 else 0.0
```

**사용 시기:**
- 검색 순위 품질 종합 평가
- 학술 논문에서 많이 사용

### B. 정성적 평가

#### 1. **Hit Rate (적중률)**
> 상위 K개 중 최소 1개라도 관련 문서가 있는 비율

```python
def hit_rate(queries_results, k):
    hits = 0
    for retrieved, relevant in queries_results:
        if any(doc in relevant for doc in retrieved[:k]):
            hits += 1

    return hits / len(queries_results)
```

#### 2. **Context Relevance (문맥 관련성)**
> 검색된 문서가 질문과 실제로 관련 있는지 판단

```python
# LLM을 사용한 평가
def evaluate_context_relevance(query, context, llm):
    """
    Args:
        query: 사용자 질문
        context: 검색된 문서
        llm: 평가용 LLM
    """
    prompt = f"""
    Query: {query}
    Context: {context}

    Does this context help answer the query?
    Answer with: RELEVANT or NOT_RELEVANT
    """

    response = llm.generate(prompt)
    return "RELEVANT" in response
```

---

## 2️⃣ Generation 평가 (생성 품질)

### A. 정량적 메트릭

#### 1. **BLEU / ROUGE** (참조 답변이 있을 때)
> 정답과 생성된 답변의 단어 일치도

```python
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def calculate_bleu(reference, generated):
    """
    Args:
        reference: 정답 답변 (리스트)
        generated: 생성된 답변 (문자열)
    """
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()

    return sentence_bleu(reference_tokens, generated_tokens)

def calculate_rouge(reference, generated):
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)[0]

    return {
        'rouge-1': scores['rouge-1']['f'],  # Unigram F1
        'rouge-2': scores['rouge-2']['f'],  # Bigram F1
        'rouge-l': scores['rouge-l']['f'],  # Longest common subsequence
    }
```

**한계:**
- 표현이 다르지만 의미가 같은 경우 낮은 점수
- 한국어에서는 형태소 분석 필요

#### 2. **BERTScore** ⭐ 추천 (한국어)
> 의미 유사도 기반 평가

```python
from bert_score import score

def calculate_bertscore(references, candidates):
    """
    Args:
        references: 정답 리스트
        candidates: 생성 답변 리스트
    """
    P, R, F1 = score(
        candidates,
        references,
        lang="ko",  # 한국어
        model_type="bert-base-multilingual-cased"
    )

    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }
```

**장점:**
- 의미적 유사도 측정
- 한국어 지원 우수

### B. RAG 특화 메트릭 ⭐⭐⭐

#### 1. **Faithfulness (충실도)** - 가장 중요!
> 답변이 검색된 문서(context)에 기반하는지

```python
def evaluate_faithfulness(question, context, answer, llm):
    """
    답변이 context에서 나온 정보만 사용했는지 평가

    Hallucination 방지!
    """
    prompt = f"""
    Question: {question}
    Context: {context}
    Answer: {answer}

    Does the answer only use information from the context?
    Score from 1-5 (5 = completely faithful, 1 = hallucination)
    """

    response = llm.generate(prompt)
    # Parse score from response
    return parse_score(response)
```

**왜 중요한가:**
- RAG의 핵심 = 검색한 정보만 사용
- Hallucination (환각) 방지

#### 2. **Answer Relevance (답변 관련성)**
> 답변이 질문에 얼마나 적절한지

```python
def evaluate_answer_relevance(question, answer, llm):
    """
    답변이 질문에 직접적으로 대답하는지
    """
    prompt = f"""
    Question: {question}
    Answer: {answer}

    Does this answer directly address the question?
    Score from 1-5 (5 = perfect answer, 1 = irrelevant)
    """

    return parse_score(llm.generate(prompt))
```

#### 3. **Context Precision (문맥 정밀도)**
> 검색된 문서가 모두 유용한지

```python
def evaluate_context_precision(question, contexts, answer, llm):
    """
    검색된 context들이 답변에 실제로 사용되었는지
    """
    useful_contexts = 0

    for ctx in contexts:
        prompt = f"""
        Question: {question}
        Context: {ctx}
        Answer: {answer}

        Was this context useful for generating the answer?
        Answer: YES or NO
        """

        if "YES" in llm.generate(prompt):
            useful_contexts += 1

    return useful_contexts / len(contexts)
```

---

## 3️⃣ 실전 평가 프레임워크

### RAGAS (RAG Assessment) ⭐ 추천

가장 널리 사용되는 RAG 평가 라이브러리

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# 평가 데이터셋 준비
eval_dataset = {
    'question': [
        "제천 시티투어는 어떻게 예약하나요?",
        "의림지는 어디에 있나요?",
        ...
    ],
    'answer': [
        "citytour.jecheon.go.kr에서 예약하거나...",
        "의림지는 제천시 모산동에 위치합니다...",
        ...
    ],
    'contexts': [
        ["제천 시티투어\n예약안내\ncitytour.jecheon.go.kr..."],
        ["의림지\n위치: 충청북도 제천시 모산동..."],
        ...
    ],
    'ground_truth': [  # Optional
        "공식 홈페이지나 전화로 예약",
        "제천시 모산동",
        ...
    ]
}

# 평가 실행
result = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,          # 답변이 context에 충실한가
        answer_relevancy,      # 답변이 질문에 관련있는가
        context_recall,        # 필요한 정보를 검색했는가
        context_precision,     # 검색된 정보가 유용한가
    ],
)

print(result)
# {
#   'faithfulness': 0.92,
#   'answer_relevancy': 0.88,
#   'context_recall': 0.85,
#   'context_precision': 0.90
# }
```

---

## 4️⃣ 제천 프로젝트 평가 전략

### 평가 데이터셋 구축

#### A. 질문-답변 쌍 생성

```python
# 1. GPT/Claude로 질문 생성
from openai import OpenAI

client = OpenAI()

def generate_qa_pairs(context_chunk):
    """
    PDF chunk에서 질문-답변 쌍 생성
    """
    prompt = f"""
    다음 제천시 관광 정보를 읽고, 3개의 질문-답변 쌍을 생성하세요.

    정보:
    {context_chunk}

    형식:
    Q1: [질문]
    A1: [답변]
    Q2: [질문]
    A2: [답변]
    Q3: [질문]
    A3: [답변]
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return parse_qa_pairs(response.choices[0].message.content)

# 2. 수동으로 검증 및 정제
# 3. train/test 분할 (80/20)
```

#### B. 평가 질문 예시 (제천 관광)

```python
eval_questions = [
    # 단순 사실 질문
    "제천 시티투어 요금은 얼마인가요?",
    "의림지 박물관은 언제 휴무인가요?",
    "청풍호반 케이블카 운영시간은?",

    # 비교 질문
    "제천의 대표 축제는 무엇인가요?",
    "가족 여행에 추천하는 코스는?",

    # 복합 질문
    "제천에서 1박 2일 여행 계획을 세워주세요",
    "겨울에 제천에서 할 수 있는 활동은?",

    # 추론 질문
    "비가 오는 날 제천에서 갈 만한 곳은?",
    "어린이와 함께 가기 좋은 관광지는?",
]
```

### 평가 메트릭 선정 (과제용)

```python
# Retrieval 평가
retrieval_metrics = {
    'MRR': mean_reciprocal_rank,      # 첫 관련 문서 순위
    'Recall@3': lambda: recall_at_k(k=3),  # 상위 3개 재현율
    'Hit_Rate@5': lambda: hit_rate(k=5),   # 상위 5개 적중률
}

# Generation 평가
generation_metrics = {
    'BERTScore_F1': calculate_bertscore,   # 의미 유사도
    'BLEU': calculate_bleu,                # 단어 일치도 (참고용)
}

# RAG 통합 평가 (RAGAS)
rag_metrics = {
    'Faithfulness': faithfulness,          # 충실도 ⭐
    'Answer_Relevancy': answer_relevancy,  # 답변 관련성 ⭐
    'Context_Precision': context_precision,# 검색 정밀도
}
```

### Baseline vs Fine-tuned 비교

```python
import pandas as pd

# 평가 실행
baseline_results = evaluate_model(baseline_model, eval_dataset)
finetuned_results = evaluate_model(finetuned_model, eval_dataset)

# 비교표 생성
comparison = pd.DataFrame({
    'Metric': list(rag_metrics.keys()),
    'Baseline': [baseline_results[m] for m in rag_metrics],
    'Fine-tuned': [finetuned_results[m] for m in rag_metrics],
    'Improvement': [
        (finetuned_results[m] - baseline_results[m]) / baseline_results[m] * 100
        for m in rag_metrics
    ]
})

print(comparison)
#          Metric  Baseline  Fine-tuned  Improvement
# 0  Faithfulness      0.72        0.89        23.6%
# 1  Answer_Rel...     0.68        0.85        25.0%
# 2  Context_Pre...    0.75        0.88        17.3%
```

---

## 5️⃣ 정성적 평가 (필수!)

### A. 예시 기반 비교 (3+ examples)

```markdown
### Example 1: 시티투어 예약

**질문:** 제천 시티투어는 어떻게 예약하나요?

**검색된 Context:**
제천 시티투어
예약안내: citytour.jecheon.go.kr
전화: 043-647-2121

**Baseline 답변:**
시티투어는 05번으로 예약하세요.
→ ❌ Hallucination (페이지 번호를 전화번호로 오인)

**Fine-tuned 답변:**
제천 시티투어는 공식 홈페이지(citytour.jecheon.go.kr)나
전화(043-647-2121)로 예약하실 수 있습니다.
→ ✅ 정확하고 완전한 답변

**평가:**
- Faithfulness: Baseline 2/5, Fine-tuned 5/5
- Relevancy: Baseline 3/5, Fine-tuned 5/5
```

### B. 실패 사례 분석

```markdown
### Failure Case 1: 복합 질문

**질문:** 1박 2일 제천 여행 코스 추천해주세요

**문제:** 여러 chunk에서 정보를 종합해야 함

**개선 방안:**
- Retrieval: top_k를 3→5로 증가
- Generation: 더 긴 context window 사용
- Re-ranking 추가
```

---

## 6️⃣ 구현 순서

### Step 1: 평가 데이터셋 생성
```bash
python scripts/generate_eval_dataset.py
# → data/eval/test_set.json
```

### Step 2: Baseline 평가
```bash
python scripts/evaluate_rag.py --model baseline
# → results/baseline_evaluation.json
```

### Step 3: Fine-tuned 평가
```bash
python scripts/evaluate_rag.py --model finetuned
# → results/finetuned_evaluation.json
```

### Step 4: 비교 리포트 생성
```bash
python scripts/generate_comparison_report.py
# → results/comparison_report.pdf
```

---

## 7️⃣ 추천 메트릭 조합

### 최소 구성 (시간 부족 시)
```python
metrics = {
    'Retrieval': ['MRR', 'Recall@3'],
    'Generation': ['BERTScore'],
    'RAG': ['Faithfulness', 'Answer_Relevancy'],
    'Qualitative': ['3 Examples', 'Failure Analysis']
}
```

### 권장 구성 ⭐
```python
metrics = {
    'Retrieval': ['MRR', 'Recall@3', 'Hit_Rate@5'],
    'Generation': ['BERTScore', 'BLEU'],
    'RAG': ['Faithfulness', 'Answer_Relevancy', 'Context_Precision'],
    'Qualitative': ['5 Examples', 'Failure Analysis', 'User Study (Optional)']
}
```

---

## 📚 참고 자료

- **RAGAS 공식 문서:** https://docs.ragas.io/
- **참조 논문:** "RAGAS: Automated Evaluation of RAG" (2023)
- **한국어 RAG 벤치마크:** https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO

---

## ✅ 체크리스트

- [ ] 평가 데이터셋 생성 (50-100 Q&A)
- [ ] Retrieval 메트릭 계산 (MRR, Recall@K)
- [ ] Generation 메트릭 계산 (BERTScore)
- [ ] RAG 메트릭 계산 (Faithfulness, Relevancy)
- [ ] Baseline vs Fine-tuned 비교표
- [ ] 정성적 예시 분석 (3+)
- [ ] 실패 사례 분석
- [ ] 시각화 (그래프, 차트)
- [ ] 리포트 작성

---

**다음 단계: 평가 데이터셋 생성 스크립트 작성**
