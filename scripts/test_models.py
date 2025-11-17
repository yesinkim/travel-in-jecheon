#!/usr/bin/env python3
"""
Model Comparison Script
모든 모델을 테스트하고 결과를 비교하는 스크립트

Usage:
    python test_models.py --models Qwen2.5-7B EXAONE-3.5-7.8B
    python test_models.py --all  # 모든 모델 테스트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import pandas as pd
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# 디렉토리 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 질문 리스트
SIMPLE_QUESTIONS = [
    "제천은 어느 도에 소속되어 있나요?",
    "K-pop을 대표하는 아티스트를 알려주세요.",
    "오늘 날씨는 어때?",
    "세종대왕은 어떤 업적을 남겼나요?",
    "삼국 시대의 세 나라는 각각 어떤 특징이 있었나요?"
]

# 모델 설정
MODEL_CONFIGS = {
    "Kanana-2.1B": {
        "name": "kakaocorp/kanana-nano-2.1b-instruct",
        "system_prompt": "You are a helpful AI assistant developed by Kakao.",
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": None,
        "use_eos_token_id": False
    },
    "EXAONE-3.5-7.8B": {
        "name": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "system_prompt": "You are EXAONE model from LG AI Research, a helpful assistant.",
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "use_eos_token_id": True
    },
    "Qwen2.5-7B": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "dtype": "auto",
        "trust_remote_code": None,
        "device_map": "auto",
        "use_eos_token_id": False
    },
    "EXAONE-4.0-1.2B": {
        "name": "LGAI-EXAONE/EXAONE-4.0-1.2B",
        "system_prompt": None,
        "dtype": "bfloat16",
        "trust_remote_code": None,
        "device_map": "auto",
        "use_eos_token_id": False
    }
}


def test_model_on_questions(
    model_key: str,
    questions: List[str],
    max_new_tokens: int = 256,
    device: str = "cuda"
) -> Dict[str, List[str]]:
    """
    특정 모델로 여러 질문을 테스트하고 결과 반환

    Args:
        model_key: MODEL_CONFIGS의 키
        questions: 테스트할 질문 리스트
        max_new_tokens: 생성할 최대 토큰 수
        device: 사용할 디바이스

    Returns:
        질문과 답변이 포함된 딕셔너리
    """
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Testing {model_key} ({config['name']})")
    print(f"{'='*60}\n")

    # 모델 및 토크나이저 로드
    print("Loading model...")

    # 모델 로드 파라미터 구성
    model_kwargs = {
        "torch_dtype": config['dtype']
    }

    # trust_remote_code 설정 (None이 아닐 때만 추가)
    if config['trust_remote_code'] is not None:
        model_kwargs["trust_remote_code"] = config['trust_remote_code']

    # device_map 설정 (Kanana는 None, 나머지는 "auto")
    if config['device_map'] is not None:
        model_kwargs["device_map"] = config['device_map']

    model = AutoModelForCausalLM.from_pretrained(
        config['name'],
        **model_kwargs
    )

    # Kanana는 .to("cuda") 사용
    if config['device_map'] is None:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config['name'])
    model.eval()

    results = {
        "model": model_key,
        "questions": [],
        "answers": [],
        "timestamps": []
    }

    # 각 질문에 대해 테스트
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]: {question}")

        # 메시지 구성
        if config['system_prompt']:
            messages = [
                {"role": "system", "content": config['system_prompt']},
                {"role": "user", "content": question}
            ]
        else:
            messages = [{"role": "user", "content": question}]

        # 입력 준비
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # 적절한 디바이스로 이동
        if config['device_map'] is None:
            input_ids = input_ids.to(device)
        elif "EXAONE" in model_key:
            input_ids = input_ids.to(device)
        else:
            input_ids = input_ids.to(model.device)

        # 생성 파라미터 구성
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False
        }

        # EXAONE-3.5-7.8B는 eos_token_id 사용
        if config['use_eos_token_id']:
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

        # 생성
        start_time = datetime.now()
        with torch.no_grad():
            output = model.generate(input_ids, **generate_kwargs)
        end_time = datetime.now()

        # 디코딩
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # 결과 저장
        results["questions"].append(question)
        results["answers"].append(generated_text)
        results["timestamps"].append((end_time - start_time).total_seconds())

        print(f"[Answer]: {generated_text[:200]}...")
        print(f"[Time]: {results['timestamps'][-1]:.2f}s")

    # 메모리 정리
    del model
    del tokenizer
    torch.cuda.empty_cache()

    print(f"\n{model_key} 테스트 완료!")

    # 모델별 중간 결과 저장 (JSON)
    save_intermediate_result(results, model_key)

    return results


def save_intermediate_result(result: Dict, model_key: str):
    """모델별 중간 결과를 JSON으로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = DATA_DIR / f"{model_key}_results_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"💾 중간 결과 저장: {filename}")


def create_comparison_df(results_list: List[Dict]) -> pd.DataFrame:
    """결과를 비교 가능한 DataFrame으로 변환"""
    comparison_data = []

    for result in results_list:
        model_name = result['model']
        for i, (question, answer, time) in enumerate(zip(
            result['questions'],
            result['answers'],
            result['timestamps']
        )):
            comparison_data.append({
                'Question #': i + 1,
                'Question': question,
                'Model': model_name,
                'Answer': answer,
                'Time (s)': round(time, 2)
            })

    return pd.DataFrame(comparison_data)


def save_comparison_results(df: pd.DataFrame, timestamp: str):
    """비교 결과를 여러 형식으로 저장"""

    # 1. CSV 저장 (전체 결과)
    csv_path = DATA_DIR / f"model_comparison_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"💾 CSV 저장: {csv_path}")

    # 2. Pivot 테이블 저장 (모델별 답변 비교)
    pivot_df = df.pivot_table(
        index=['Question #', 'Question'],
        columns='Model',
        values='Answer',
        aggfunc='first'
    ).reset_index()

    pivot_path = DATA_DIR / f"model_comparison_pivot_{timestamp}.csv"
    pivot_df.to_csv(pivot_path, index=False, encoding='utf-8-sig')
    print(f"💾 Pivot 테이블 저장: {pivot_path}")

    # 3. 평균 응답 시간 저장
    avg_times = df.groupby('Model')['Time (s)'].mean().sort_values()

    stats_df = pd.DataFrame({
        'Model': avg_times.index,
        'Average Time (s)': avg_times.values,
        'Total Questions': [len(df[df['Model'] == model]) for model in avg_times.index]
    })

    stats_path = DATA_DIR / f"model_statistics_{timestamp}.csv"
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"💾 통계 저장: {stats_path}")

    return pivot_df, stats_df


def print_comparison_summary(df: pd.DataFrame):
    """비교 결과를 콘솔에 출력"""
    print("\n" + "="*80)
    print("모델별 답변 비교")
    print("="*80)

    for question_num in df['Question #'].unique():
        question_data = df[df['Question #'] == question_num]
        question_text = question_data['Question'].iloc[0]

        print(f"\n\n📌 질문 {question_num}: {question_text}")
        print("-" * 80)

        for _, row in question_data.iterrows():
            print(f"\n🤖 {row['Model']} ({row['Time (s)']}초):")
            print(f"{row['Answer']}\n")

    # 평균 응답 시간 비교
    print("\n" + "="*80)
    print("모델별 평균 응답 시간")
    print("="*80)
    avg_times = df.groupby('Model')['Time (s)'].mean().sort_values()
    print(avg_times)


def create_visualization(df: pd.DataFrame, timestamp: str):
    """결과 시각화"""

    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 모델별 평균 응답 시간 막대 그래프
    fig, ax = plt.subplots(figsize=(10, 6))

    avg_times = df.groupby('Model')['Time (s)'].mean().sort_values()
    avg_times.plot(kind='barh', ax=ax, color='steelblue')

    ax.set_xlabel('평균 응답 시간 (초)', fontsize=12)
    ax.set_ylabel('모델', fontsize=12)
    ax.set_title('모델별 평균 응답 시간 비교', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # 저장
    plot_path = RESULTS_DIR / f'model_response_time_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 그래프 저장: {plot_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='모델 비교 테스트 스크립트')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(MODEL_CONFIGS.keys()),
        help='테스트할 모델 선택'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='모든 모델 테스트'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='생성할 최대 토큰 수 (기본: 256)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='사용할 디바이스 (기본: cuda)'
    )

    args = parser.parse_args()

    # CUDA 사용 가능 확인
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        args.device = 'cpu'

    # 테스트할 모델 결정
    if args.all:
        models_to_test = list(MODEL_CONFIGS.keys())
    elif args.models:
        models_to_test = args.models
    else:
        # 기본값: Qwen2.5-7B만 테스트
        models_to_test = ["Qwen2.5-7B"]
        print("⚠️  모델이 지정되지 않았습니다. Qwen2.5-7B만 테스트합니다.")
        print("   다른 모델을 테스트하려면 --models 또는 --all 옵션을 사용하세요.\n")

    print("="*60)
    print("모델 비교 테스트 시작")
    print("="*60)
    print(f"테스트할 모델: {', '.join(models_to_test)}")
    print(f"질문 개수: {len(SIMPLE_QUESTIONS)}")
    print(f"최대 토큰: {args.max_tokens}")
    print(f"디바이스: {args.device}")
    print("="*60)

    # 타임스탬프 생성 (모든 결과에 동일한 타임스탬프 사용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 모든 모델 테스트
    all_results = []

    for model_key in models_to_test:
        try:
            result = test_model_on_questions(
                model_key=model_key,
                questions=SIMPLE_QUESTIONS,
                max_new_tokens=args.max_tokens,
                device=args.device
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error testing {model_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\n❌ 테스트된 모델이 없습니다.")
        return

    print(f"\n✅ 총 {len(all_results)}개 모델 테스트 완료!")

    # DataFrame 생성
    df_comparison = create_comparison_df(all_results)

    # 결과 출력
    print_comparison_summary(df_comparison)

    # 결과 저장
    print("\n" + "="*60)
    print("결과 저장 중...")
    print("="*60)

    pivot_df, stats_df = save_comparison_results(df_comparison, timestamp)

    # 시각화 생성
    try:
        create_visualization(df_comparison, timestamp)
    except Exception as e:
        print(f"⚠️  시각화 생성 실패: {str(e)}")

    # 최종 통계 출력
    print("\n" + "="*60)
    print("최종 통계")
    print("="*60)
    print(stats_df.to_string(index=False))

    print("\n✅ 모든 작업 완료!")
    print(f"\n결과 파일 위치:")
    print(f"  - 데이터: {DATA_DIR}")
    print(f"  - 그래프: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
