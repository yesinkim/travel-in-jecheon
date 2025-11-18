# 학습 중 평가 가이드 (Training Evaluation Guide)

## 목차
1. [평가 지표 해석](#평가-지표-해석)
2. [학습 곡선 분석](#학습-곡선-분석)
3. [문제 상황 대응](#문제-상황-대응)
4. [체크포인트 선택](#체크포인트-선택)

---

## 평가 지표 해석

### 기본 지표

| 지표 | 설명 | 좋은 값 | 나쁜 신호 |
|------|------|---------|----------|
| `train_loss` | 학습 데이터 손실 | 감소 추세 | 증가 or 정체 |
| `eval_loss` | 검증 데이터 손실 | 감소 추세 | 증가 (overfitting) |
| `perplexity` | 언어 모델 성능 | 낮을수록 좋음 | 높거나 증가 |
| `learning_rate` | 현재 학습률 | 점진적 감소 | 너무 빠른 변화 |

### Loss 값 해석 (Qwen2.5-7B 기준)

```
초기 loss (epoch 0):     2.0 - 3.0  (정상)
중간 loss (epoch 1-2):   1.0 - 2.0  (학습 중)
최종 loss (epoch 3):     0.5 - 1.5  (목표)

⚠️ 주의:
- Loss < 0.3: Overfitting 가능성 (너무 낮음)
- Loss > 3.0: 학습 안 됨 (초기 상태)
- Eval loss > Train loss + 0.5: Overfitting
```

### Perplexity 해석

```python
Perplexity = exp(loss)

예시:
- Loss 2.0 → Perplexity 7.4  (초기)
- Loss 1.0 → Perplexity 2.7  (좋음)
- Loss 0.5 → Perplexity 1.6  (매우 좋음)

의미: "모델이 다음 토큰을 예측할 때 평균적으로 N개 후보 중 고민"
낮을수록 확신도가 높음.
```

---

## 학습 곡선 분석

### 1. 정상 학습 (Ideal)

```
Step    Train Loss    Eval Loss    판단
----    ----------    ---------    ----
10      2.50          2.55         정상 시작
50      2.10          2.15         ✅ 정상
100     1.70          1.80         ✅ 정상
150     1.40          1.50         ✅ 정상
200     1.15          1.25         ✅ 정상
250     0.95          1.10         ✅ 계속 진행

특징:
- 두 loss 모두 감소
- Gap이 일정 (0.05 - 0.15)
- 안정적인 감소 추세
```

### 2. Overfitting 시작

```
Step    Train Loss    Eval Loss    판단
----    ----------    ---------    ----
10      2.50          2.55         정상
50      2.10          2.15         ✅ 정상
100     1.70          1.80         ✅ 정상
150     1.40          1.55         ⚠️ gap 증가
200     1.15          1.60         ❌ eval 증가
250     0.90          1.75         ❌ overfitting!

대응:
→ Step 100-150 checkpoint 사용
→ Regularization 추가 (dropout, weight decay)
→ Early stopping 적용
```

### 3. Underfitting

```
Step    Train Loss    Eval Loss    판단
----    ----------    ---------    ----
10      2.50          2.55         정상
50      2.45          2.50         ⚠️ 느린 감소
100     2.40          2.45         ❌ 거의 변화 없음
150     2.35          2.42         ❌ underfitting

대응:
→ Learning rate 증가 (2e-5 → 5e-5)
→ 더 많은 epoch
→ 모델 크기 증가 (3B → 7B)
→ Batch size 조정
```

### 4. Learning Rate 문제

**너무 높음:**
```
Step    Train Loss    판단
----    ----------    ----
10      2.50          정상
50      3.10          ❌ 증가!
100     2.80          불안정
150     3.50          ❌ 발산

대응: LR 1/10로 줄이기 (2e-5 → 2e-6)
```

**너무 낮음:**
```
Step    Train Loss    판단
----    ----------    ----
10      2.50          정상
50      2.49          거의 변화 없음
100     2.48          ❌ 너무 느림
150     2.47          시간 낭비

대응: LR 2-5배 증가 (2e-5 → 5e-5)
```

---

## 문제 상황 대응

### 상황 1: Loss가 NaN

```python
문제: train_loss = NaN

원인:
1. Learning rate 너무 높음
2. Gradient explosion
3. 데이터에 inf/nan 값 포함

해결:
✅ Learning rate 1/10로 줄이기
✅ Gradient clipping 추가:
   training_args = TrainingArguments(
       max_grad_norm=1.0,  # 추가
   )
✅ 데이터 검증:
   dataset.filter(lambda x: x['text'] is not None)
```

### 상황 2: GPU Out of Memory

```python
문제: CUDA out of memory

해결:
✅ Batch size 줄이기: 4 → 2 → 1
✅ Gradient accumulation 증가:
   gradient_accumulation_steps=8  # effective batch = 1*8=8
✅ Gradient checkpointing 활성화:
   gradient_checkpointing=True
✅ Mixed precision 사용:
   bf16=True  # or fp16=True
✅ Max length 줄이기:
   max_length=512 → 256
```

### 상황 3: 학습이 너무 느림

```python
문제: 1 epoch에 5시간 이상

해결:
✅ Batch size 증가 (메모리 허용 시)
✅ Dataloader workers 증가:
   dataloader_num_workers=4
✅ Pin memory 활성화:
   dataloader_pin_memory=True
✅ 불필요한 evaluation 줄이기:
   eval_steps=100  (50 → 100)
✅ 로깅 줄이기:
   logging_steps=50  (10 → 50)
```

### 상황 4: Validation Loss만 증가

```python
문제: train_loss 감소, eval_loss 증가

원인: Overfitting (훈련 데이터 암기)

해결:
✅ Early stopping 사용:
   load_best_model_at_end=True
✅ Dropout 추가 (LoRA 사용 시):
   lora_dropout=0.1
✅ Weight decay 증가:
   weight_decay=0.01  (0.0 → 0.01)
✅ Epoch 수 줄이기:
   num_epochs=2  (3 → 2)
✅ 데이터 증강 (if possible)
```

---

## 체크포인트 선택

### 저장된 체크포인트 확인

```bash
# 학습 후 체크포인트 목록
ls -lh results/qwen-7b-jecheon/

출력 예시:
checkpoint-50/      # Step 50
checkpoint-100/     # Step 100
checkpoint-150/     # Step 150  ← Best (eval_loss 최저)
checkpoint-200/     # Step 200
final_model/        # 최종 모델
```

### Best Checkpoint 찾기

```python
# trainer_state.json에서 best checkpoint 확인
import json

with open("results/qwen-7b-jecheon/trainer_state.json") as f:
    state = json.load(f)

# Best checkpoint 경로
best_checkpoint = state["best_model_checkpoint"]
print(f"Best: {best_checkpoint}")

# 각 checkpoint의 eval_loss
for log in state["log_history"]:
    if "eval_loss" in log:
        print(f"Step {log['step']}: eval_loss = {log['eval_loss']:.4f}")
```

### 수동 선택 기준

```python
체크포인트 선택 우선순위:

1. Lowest eval_loss
   → 가장 일반화 성능 좋음

2. Train/Eval loss gap 가장 작음
   → 가장 안정적

3. Sample 예측 품질이 가장 좋음
   → 실제 사용 성능 고려

예시:
checkpoint-100: eval_loss=1.2, gap=0.1, 예측품질=보통
checkpoint-150: eval_loss=1.1, gap=0.2, 예측품질=좋음  ← 선택
checkpoint-200: eval_loss=1.3, gap=0.4, 예측품질=매우좋음 (overfit)
```

---

## 실전 체크리스트

### 학습 시작 전

```bash
✅ 데이터셋 크기 확인
   - Train: 100+ samples
   - Validation: 10-20 samples

✅ GPU 메모리 확인
   nvidia-smi

✅ 설정 확인
   - Learning rate: 2e-5
   - Batch size: 4 (또는 메모리에 맞게)
   - Eval steps: 50
   - Max length: 512

✅ 로깅 설정
   - Wandb 로그인 확인: wandb login
   - 프로젝트 이름 설정
```

### 학습 시작 후 (10 steps)

```bash
✅ Loss 감소 시작
   Step 1:  2.5
   Step 10: 2.3 ✅

✅ 메모리 사용량 안정
   nvidia-smi | grep python

✅ No errors in logs
   tail -f nohup.out
```

### 첫 Evaluation (50 steps)

```bash
✅ Eval loss 확인
   eval_loss < 3.0 ✅

✅ Train/Eval gap 확인
   |train_loss - eval_loss| < 0.5 ✅

✅ Sample 예측 확인
   답변이 의미 있는 문장인가? ✅
```

### 매 Evaluation

```bash
✅ Loss 추세
   eval_loss 계속 감소? ✅
   train/eval gap 증가하지 않음? ✅

✅ 샘플 품질
   답변이 점점 개선되는가? ✅

✅ Checkpoint 저장
   새 checkpoint 생성되었는가? ✅
```

### 학습 종료 후

```bash
✅ Best checkpoint 식별
   cat results/*/trainer_state.json | grep best_model

✅ 최종 평가
   python scripts/evaluate.py --checkpoint results/.../checkpoint-XXX

✅ 샘플 테스트
   질문 3-5개로 실제 답변 품질 확인

✅ 모델 저장
   업로드 전 최종 테스트
```

---

## Weights & Biases 활용

### 설치 및 로그인

```bash
pip install wandb
wandb login
# API key 입력: https://wandb.ai/authorize
```

### 실시간 모니터링

```bash
# 학습 시작 후 브라우저에서 확인
https://wandb.ai/<username>/jecheon-rag-finetuning

확인 항목:
📊 Charts:
   - train/loss
   - eval/loss
   - learning_rate

📈 비교:
   - 여러 실험 동시 비교
   - Hyperparameter 영향 분석

💾 Artifacts:
   - Best checkpoint 자동 저장
   - 모델 버전 관리
```

### 중요 그래프

```python
1. Loss 그래프 (가장 중요!)
   Y축: loss
   X축: step
   ✅ 감소 추세
   ❌ 증가 or 정체

2. Learning Rate Schedule
   Warmup → Constant/Decay 확인

3. Gradient Norm
   너무 크면 (>10) → clipping 필요

4. Train/Eval Loss 비교
   두 선이 가까워야 함
   Gap 벌어지면 overfitting
```

---

## TensorBoard 활용 (대안)

```bash
# 학습 중 별도 터미널에서
tensorboard --logdir ./results

# 브라우저: http://localhost:6006

탭별 확인 사항:
📊 SCALARS:
   - train/loss
   - eval/loss

📈 GRAPHS:
   - 모델 구조 시각화

📁 HPARAMS:
   - Hyperparameter 비교
```

---

## 요약: 좋은 학습의 신호

```
✅ Train loss 꾸준히 감소
✅ Eval loss도 함께 감소
✅ Train/Eval gap < 0.5
✅ 샘플 예측 점점 개선
✅ No NaN, no OOM errors
✅ Perplexity < 3.0 (final)
✅ Loss 안정적 (진동 없음)

이 7가지가 모두 만족되면 성공적인 학습!
```

---

## 참고 자료

- [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Weights & Biases Guide](https://docs.wandb.ai/)
- [Understanding Loss Curves](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
