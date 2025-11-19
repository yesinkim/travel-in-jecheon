# 제천시 관광정보 RAG 시스템 구축 및 Fine-tuning 결과 보고서

**작성일:** 2025-11-19
**작성자:** 김예신

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [RAG 데이터셋 구축 과정](#2-rag-데이터셋-구축-과정)
3. [Baseline 모델 평가](#3-baseline-모델-평가)
4. [RAG Fine-tuning 수행](#4-rag-fine-tuning-수행)
5. [Fine-tuned 모델 평가 및 비교](#5-fine-tuned-모델-평가-및-비교)
6. [구현 코드 설명](#6-구현-코드-설명)
7. [결론 및 향후 개선 방안](#7-결론-및-향후-개선-방안)

---

## 1. 프로젝트 개요

### 1.1 과제 목표

본 프로젝트는 제천시 관광정보 책자(PDF)를 기반으로 **RAG(Retrieval-Augmented Generation) 시스템**을 구축하고, 이를 활용해 **LLM Fine-tuning**을 수행하여 도메인 특화 답변 성능을 개선하는 것을 목표로 합니다.

**핵심 목표:**
1. ✅ 비정형 PDF 문서를 고품질 RAG용 데이터셋으로 변환 (파싱, 청킹, QA 생성)
2. ✅ Baseline LLM 선정 및 RAG 파이프라인 성능 평가
3. ✅ RAG 태스크(Context 인지 및 답변 생성)에 특화된 Fine-tuning 수행
4. ✅ **Fine-tuning 전후 성능 정량/정성 비교 분석 (실패 원인 규명)**
5. ✅ Hugging Face에 데이터셋 및 모델 업로드

### 1.2 접근 방식

데이터 중심(Data-Centric) 접근 방식을 채택하여 전처리 과정에 리소스의 50% 이상을 투입했습니다.

1. **데이터 구축**: PDF 파싱(OCR 보정) → 의미 단위 청킹 → QA 및 Distractor(오답 문서) 생성
2. **모델 선정**: 리더보드 및 자체 정성 평가를 통해 베이스 모델(`Kanana-1.5-8B`) 선정
3. **Baseline 평가**: RAG 파이프라인 구축 후 성능 측정 (ROUGE, BERTScore, LLM Judge)
4. **Fine-tuning**: QLoRA 기법을 적용하여 데이터 효율적인 학습 수행
5. **비교 분석**: Baseline과 Fine-tuned 모델의 실패 패턴 분석

### 1.3 사용 기술 스택

| 카테고리 | 기술 스택 |
|----------|-----------|
| **언어** | Python 3.11 |
| **LLM 프레임워크** | Transformers 4.45, PEFT (QLoRA) |
| **RAG 프레임워크** | LangChain, FAISS |
| **평가 도구** | ROUGE, BERTScore, Gemini 2.5 Pro (LLM Judge) |
| **데이터 처리** | Gemini 2.5 Pro (Vision), Pandas |
| **인프라** | RunPod (A100 SXM 100G) |

---

## 2. RAG 데이터셋 구축 과정

RAG 시스템의 성능은 데이터 품질에 종속된다고 판단하여, 정교한 전처리 파이프라인을 구축했습니다.

### 2.1 PDF 파싱 및 보정: Gemini Vision 활용

기존 OCR 도구(PyMuPDF, PyZerox)들의 한계(다단 레이아웃 깨짐, 미세한 텍스트 오류)를 극복하기 위해 **Gemini 2.5 Pro의 Vision 기능**을 활용했습니다.

*   **방법**: PDF 이미지를 입력받아 Markdown으로 변환하되, "주소, 전화번호, 가격 정보의 정확성"을 강제하는 프롬프트 사용.
*   **결과**: 원본 대조 결과 주소 및 연락처 오류율 0%를 달성하며 구조(표, 리스트)가 완벽히 보존된 Markdown 데이터 확보.

### 2.2 문서 청킹 (Chunking)

*   **전략**: 의미론적 경계 유지 (Semantic Chunking)
*   **구현**: 목차(TOC)와 헤더(`###`)를 기준으로 분할 후, 300~2000자 사이로 크기 조정.
*   **결과**: 총 42개의 고품질 문서 청크 생성.

### 2.3 QA 및 Distractor 생성

모델이 단순 암기가 아닌 '문맥 추론(Reasoning)'을 학습하도록 **Finetune-RAG** 방법론을 적용했습니다.
RAG의 중요한 모델인 Embedding model과 LLM model중 LLM을 학습해, Embedding model에 따른 Retrieval 결과에 큰 영향을 받지 않도록, 강건성을 학습시키는 것이 목표였습니다.

따라 가설은 "Finetune-RAG"에 따른 학습을 하면 강건성이 높아진다. 였으나, 가설 검증에 실패하였습니다.


*   **QA 생성**: Gemini를 활용하여 사실(Factual), 설명(Descriptive), 비교(Comparison), 정보 없음(No-answer) 등 다양한 유형의 질문 125개 생성.
*   **Distractor(오답 문서) 추가**: 각 질문당 **정답 문서 1개 + 오답 문서 2개**를 매칭.
    *   *Hard Negative*: 동일 카테고리 내 유사 문서 (혼동 유발)
    *   *Easy Negative*: 다른 카테고리 문서

### 2.4 데이터 분할

*   **총 데이터**: 125개 (Train 105 / Val 10 / Test 10)
*   **방식**: Stratified Sampling (질문 유형 및 난이도 비율 유지)

---

## 3. Baseline 모델 평가

### 3.1 모델 선정 및 평가 지표

*   **모델**: `Kanana-1.5-8B` (한국어 특화, 10B 이하 경량 모델)
*   **평가 지표**:
    *   **Automatic**: ROUGE-L (어휘 일치), BERTScore (의미 유사도)
    *   **LLM Judge**: Accuracy, Helpfulness (Gemini 2.5 Pro 활용)

### 3.2 Baseline 성능 결과

RAG 파이프라인(Retriever: `intfloat/multilingual-e5-large`, Top-K=3) 평가 결과입니다.

| 메트릭 | 점수 | 해석 |
|--------|------|------|
| **Accuracy Rate** | **0%** (0/10) | 모든 질문에 대해 오답 또는 동문서답 |
| **Helpfulness** | 1.0 / 10 | 사용자에게 전혀 도움이 되지 않음 |
| **ROUGE-L** | 0.101 | 정답과 어휘적 일치도 매우 낮음 |

**분석**: Baseline 모델은 사용자의 질문을 입력으로 받아 답변하는 것이 아니라, **Context를 보고 스스로 새로운 질문과 답변 예시를 생성(Task Hallucination)**하는 경향을 보였습니다.

---

## 4. RAG Fine-tuning 수행

### 4.1 Fine-tuning 전략

*   **방법론**: QLoRA (Quantized Low-Rank Adaptation)
*   **참고**: "Finetune-RAG: Improving Retrieval Augmented Generation" (2024) 논문의 방법론(정답+오답 문서 혼합 학습)을 적용.
*   **학습 목표**: 검색된 문서 내에서 정답을 추출하고, Distractor를 무시하는 능력 배양.

### 4.2 하이퍼파라미터 설정

| 파라미터 | 설정값 |
|----------|--------|
| Epochs | 5 |
| Batch Size | 8 (Per device 2 * Grad Accum 4) |
| Learning Rate | 5e-5 |
| LoRA Rank | 8 |

### 4.3 학습 과정 이슈

*   **WandB 연동 실패**: 클라우드 환경(RunPod) 설정 미숙으로 인해 실시간 Loss 그래프가 누락되었습니다. (로그 확인 결과 Training Loss는 수렴하였으나, 결과적으로 과적합 발생)

---

## 5. Fine-tuned 모델 평가 및 비교

**결론부터 말씀드리면, RAG Fine-tuning은 실패했습니다.** Baseline의 문제를 해결하지 못했고, Fine-tuned 모델은 언어 모델로서의 기능이 붕괴되었습니다.

### 5.1 정량적 성능 비교

| 메트릭 | Baseline (Kanana-1.5-8B) | Fine-tuned (Rag-jecheon) | 변화 |
| :--- | :---: | :---: | :---: |
| **ROUGE-L** | 0.101 | **0.053** | ▼ 47.5% |
| **BERTScore F1** | 0.601 | **0.584** | ▼ 2.8% |
| **Accuracy** | 0% | 0% | - |

### 5.2 정성적 평가: 실패 사례 분석 (Qualitative Analysis)

두 모델의 답변 패턴을 분석하여 실패 원인을 규명했습니다.

#### [사례 1] 비교 질문 실패 (객실 수 비교)
**질문**: "HOUND 하운드호텔 제천역점이랑 제천 더월 호텔 중에 객실이 더 많은 곳은 어디인가요?"
**정답**: "...HOUND 하운드호텔(41개)이 더 많습니다."

> **🔴 Base Model 답변:**
> *질문: 제천시의 대표적인 관광지와 그 특징을 알려주세요.*
> *답변: 제천시는 충청북도 북부에 위치한 도시로... 1. 의림지... 2. 청풍문화재단지...*
>
> **분석**: 사용자의 질문("호텔 객실 비교")을 완전히 무시하고, Context를 읽은 뒤 스스로 **'관광지 추천'이라는 새로운 질문을 만들고 그에 대해 답변**했습니다. (Instruction Following 실패)

> **🔴 Fine-tuned Model 답변:**
> *Information: 제천은 산과 강이 어우러진...*
> *Information: 제천은 다양한 축제로도...*
> *Question: 제천에서 렌터카를 대여하려면... Answer: ...*
> *Question: 제천에서 맛집을... Answer: ...*
>
> **분석**: 질문에 답하는 것이 아니라, **학습 데이터셋의 구조(Information-Question-Answer)를 맹목적으로 반복 생성**했습니다. 이는 모델이 데이터의 패턴에 과도하게 과적합(Overfitting)되어 추론 능력을 상실했음을 보여줍니다.

#### [사례 2] 사실 질문 실패 (운영 시간 및 요금)
**질문**: "의림지 수리공원은 언제 운영하고 이용료는 얼마인가요?"
**정답**: "...7~8월 운영, 이용료는 무료입니다."

> **🔴 Base Model 답변:**
> *질문: 제천시의 관광 명소 중에서 가족 단위로 방문하기 좋은 곳을 추천해 주세요.*
> *답변: ... 1. 의림지 ... 2. 박달재 ...*
>
> **분석**: 역시 사용자의 질문을 무시하고, 본인이 생성하기 쉬운 일반적인 질문("가족 여행지 추천")으로 대체하여 답변했습니다.

> **🔴 Fine-tuned Model 답변:**
> *Information: 제천은 산과 강이...*
> *Information: 제천은 다양한 맛집과... (반복)*
> *Information: 제천은 관광객을 위한... (반복)*
> *Information: 제... (중단)*
>
> **분석**: `Information` 태그 뒤에 무의미한 문장들을 무한 반복(Looping)하다가 토큰 제한으로 잘렸습니다. Repetition Penalty가 작동하지 않을 정도로 모델이 망가졌습니다.

### 5.3 실패 원인 종합 분석

1.  **Baseline 모델의 부적합성 (Instruction Following 부재)**:
    *   `Kanana-1.5-8B` 모델은 RAG 프롬프트("문서를 참고하여 질문에 답하시오")를 수행하는 대신, 문서를 읽고 **FAQ를 생성하는 Completion Task로 오인**하는 경향이 강했습니다.
    *   기본 모델이 지시를 따르지 않는데, 이를 기반으로 복잡한 RAG Fine-tuning을 시도한 것이 패착이었습니다.

2.  **Fine-tuning 과적합 (Mode Collapse)**:
    *   Fine-tuned 모델은 질문에 대한 답변을 학습한 것이 아니라, **학습 데이터의 텍스트 형식(Syntax)** 자체를 암기해버렸습니다. 그 결과, 어떤 입력이 들어와도 학습 데이터와 유사한 텍스트 덩어리를 뱉어내는 '앵무새'가 되었습니다.

---

## 6. 구현 코드 설명

### 6.1 프로젝트 구조

```
goodganglabs/
├── scripts/
│   ├── 00_pdf_to_markdown.py      # Gemini Vision 기반 파싱
│   ├── 02_generate_qa_with_gemini.py # QA 데이터셋 생성
│   └── ...
├── src/
│   ├── evaluation/                # ROUGE, BERTScore, LLM Judge
│   └── rag_pipeline/              # LangChain 기반 RAG 로직
└── data/                          # 처리된 데이터셋 (Train/Val/Test)
```

### 6.2 주요 모듈

*   **데이터 전처리**: Gemini Vision API를 활용해 PDF의 레이아웃과 텍스트를 완벽하게 보존하며 Markdown으로 변환.
*   **학습**: `peft` 라이브러리의 QLoRA를 사용하여 메모리 효율적인 학습 파이프라인 구축.

---

## 7. 결론 및 향후 개선 방안

### 7.1 프로젝트 요약

본 프로젝트는 제천시 관광 정보를 위한 RAG 시스템 구축을 목표로 진행되었습니다. **데이터 구축 단계에서는 Gemini를 활용하여 고품질의 RAG 데이터셋을 확보하는 성과**를 거두었으나, **모델 선정 및 학습 단계에서는 실패**했습니다. Baseline 모델은 지시 사항을 따르지 않았고, Fine-tuned 모델은 데이터 포맷에 과적합되었습니다.

### 7.2 인사이트 (Lessons Learned)

1.  **"Garbage in, Garbage out"은 모델에도 적용된다**: 데이터가 아무리 좋아도, Baseline 모델이 기본적인 Instruction Following 능력이 없다면(Task Hallucination 발생), Fine-tuning으로 이를 교정하는 것은 매우 어렵습니다.
2.  **RAG Fine-tuning은 양날의 검**: 올바르게 수행되면 도메인 지식을 강화하지만, 잘못되면 모델의 기본 대화 능력마저 파괴(Catastrophic Forgetting)할 수 있음을 확인했습니다.
3.  **초기 정성 평가의 중요성**: 본격적인 학습 전, Baseline 모델이 RAG 프롬프트에 어떻게 반응하는지 소규모 샘플로 철저히 검증했어야 합니다.

### 7.3 향후 개선 방안

1.  **Baseline 모델 교체**: Instruction Following 성능이 검증된 `Gemma-2-9B-It` 또는 `Llama-3.1-8B-Instruct` 모델로 변경하여 재시도합니다.
2.  **프롬프트 엔지니어링 단순화**: 모델이 Task를 오해하지 않도록 RAG 프롬프트를 더욱 명확하고 단순하게(Chat Template 준수) 수정합니다.
3.  **단계별 디버깅**: Retrieval(검색) 정확도와 Generation(생성) 정확도를 분리하여 평가하는 파이프라인을 구축합니다.

---

### 부록

*   **Hugging Face Dataset**: [https://huggingface.co/datasets/bailando/travel-in-jecheon](https://huggingface.co/datasets/bailando/travel-in-jecheon)
*   **Hugging Face Model**: [https://huggingface.co/bailando/kanana-jecheon](https://huggingface.co/bailando/kanana-jecheon)