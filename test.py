import sys
import types
import torch.nn as nn
import traceback

# ==============================================================================
# [긴급 패치] 구버전 PEFT와 신버전 Transformers 호환성 문제 해결
# ==============================================================================
try:
    from transformers import modeling_layers
except ImportError:
    mock_module = types.ModuleType("transformers.modeling_layers")
    class MockGradientCheckpointingLayer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.gradient_checkpointing = False 
    mock_module.GradientCheckpointingLayer = MockGradientCheckpointingLayer
    sys.modules["transformers.modeling_layers"] = mock_module
    print("🩹 [긴급 패치 적용] transformers.modeling_layers 모킹 완료")

# ==============================================================================
# Imports
# ==============================================================================
import os
import json
import pandas as pd
import torch
from typing import List, Dict, TypedDict
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
# LangChain 임베딩 import 수정
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from langgraph.graph import StateGraph, END

# 외부 평가 모듈 Import
try:
    from src.evaluation.comprehensive_evaluator import ComprehensiveRAGEvaluator
    print("✅ 외부 평가 모듈(comprehensive_evaluator) 로드 성공")
except ImportError:
    try:
        from comprehensive_evaluator import ComprehensiveRAGEvaluator
        print("✅ 외부 평가 모듈(comprehensive_evaluator) 로드 성공 (현재 경로)")
    except ImportError:
        print("❌ 오류: 'comprehensive_evaluator.py' 파일을 찾을 수 없습니다.")
        # sys.exit(1) # 에러 나도 일단 진행하도록 주석 처리

# ==============================================================================
# 설정 (Configuration)
# ==============================================================================
load_dotenv()
# 모델 이름이 정확해야 합니다.
SELECTED_MODEL = "Rag-jecheon"  

CONFIG = {
    "NAME": SELECTED_MODEL,
    "DOC_PATH": "data/chunks/documents.jsonl", # 경로 확인 필수
    "TEST_PATH": "data/processed/test.jsonl",  # 경로 확인 필수
    "VECTOR_DB_PATH": "data/faiss_index",
    "OUTPUT_CSV": f"{SELECTED_MODEL}_model_result.csv",
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
}

# ==============================================================================
# 함수 정의
# ==============================================================================
def load_or_create_vectorstore():
    print("🔧 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(CONFIG["VECTOR_DB_PATH"]):
        print(f"📂 저장된 벡터 스토어 로드: {CONFIG['VECTOR_DB_PATH']}")
        try:
            vectorstore = FAISS.load_local(CONFIG["VECTOR_DB_PATH"], embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            print(f"⚠️ 로드 실패: {e}")

    print(f"📄 문서 로드 및 생성: {CONFIG['DOC_PATH']}")
    documents = []
    if os.path.exists(CONFIG['DOC_PATH']):
        with open(CONFIG["DOC_PATH"], 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                doc = Document(
                    page_content=data['content'],
                    metadata={'doc_id': data['doc_id'], 'title': data['title']}
                )
                documents.append(doc)
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(CONFIG["VECTOR_DB_PATH"])
        return vectorstore
    else:
        print("❌ 문서 파일이 없습니다!")
        return None

# LangGraph State
class RAGState(TypedDict):
    question: str
    context: str
    answer: str

# 모델 설정
MODEL_CONFIGS = {
    "Gemma-2-9B": {
        "name": "google/gemma-2-9b-it",
        "dtype": torch.bfloat16,
    },
    "Kanana-1.5-8B": {
        "name": "kakaocorp/kanana-1.5-8b-instruct-2505",
        "dtype": torch.float16,
    },
    "Rag-jecheon": {
        "name": "bailando/kanana-jecheon",
        "dtype": torch.bfloat16, 
    }
}

def create_rag_app(vectorstore, model_key):
    # 1. LLM 로드
    config_dict = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["Kanana-1.5-8B"])
    model_name = config_dict['name']
    
    print(f"🤖 LLM 로드 중: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if "model_type" in config_dict:
        print(f"✅ [DEBUG] 모델 타입을 '{config_dict['model_type']}' (으)로 강제 설정합니다.")
        config = AutoConfig.from_pretrained(
        model_name,
        model_type=config_dict['model_type'] # <<< 이 라인을 추가/수정!
    )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config, # 수정된 config 객체를 사용!
            torch_dtype=config_dict['dtype'],
            device_map="auto"
        )
    else:
        # 'model_type'이 지정되지 않았다면 기존 방식대로 로드합니다.
        print("✅ [DEBUG] 기본 설정으로 모델을 로드합니다.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=config_dict['dtype'],
            device_map="auto"
        )
    # === 수정 로직 끝 ===

    print("✅ LLM 로드 성공!")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        return_full_text=False,
        do_sample=True,
        temperature=0.1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. 노드 정의 (이 부분이 함수 밖으로 나오지 않고 내부에 있어야 합니다)
    def retrieve_node(state: RAGState):
        docs = retriever.invoke(state["question"])
        return {"context": docs}

    def generate_node(state: RAGState):
        # [중요] state에서 변수를 꺼내야 NameError가 안 납니다.
        question = state["question"]
        context = state["context"]
        
#         prompt = f"""당신은 제천시 관광 안내 전문가입니다. 
# 제공된 문서 내용을 바탕으로 질문에 정확하고 친절하게 답변해주세요.

# 문서 내용:
# {context}

# 질문: {question}

# 답변:"""

        from langchain import hub

        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | llm
        response = rag_chain.invoke({"question": question, "context": context})
        return {"answer": response}

    # 3. 그래프 생성
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # [중요] 컴파일된 앱을 반드시 리턴해야 합니다!
    return workflow.compile()

# ==============================================================================
# 메인 실행
# ==============================================================================
def main():
    print(f"🚀 실행 시작 (Model: {SELECTED_MODEL})")
    
    # 1. VectorStore
    vectorstore = load_or_create_vectorstore()
    if vectorstore is None:
        print("❌ 벡터 스토어 생성 실패")
        return

    # 2. Evaluator 준비
    try:
        print("⚖️ Evaluator 초기화 중...")
        doc_map = {}
        if os.path.exists(CONFIG["DOC_PATH"]):
            with open(CONFIG["DOC_PATH"], 'r', encoding='utf-8') as f:
                for line in f:
                    d = json.loads(line)
                    doc_map[d['doc_id']] = d
        
        evaluator = ComprehensiveRAGEvaluator(gemini_model="gemini-2.5-pro")
    except Exception as e:
        print(f"⚠️ Evaluator 초기화 중 경고: {e}")
        evaluator = None

    # 3. RAG App 생성
    app = create_rag_app(vectorstore, SELECTED_MODEL)
    if app is None:
        print("❌ RAG 앱 생성 실패 (app is None)")
        return

    # 4. 테스트 데이터 로드
    if not os.path.exists(CONFIG['TEST_PATH']):
        print(f"❌ 테스트 데이터 없음: {CONFIG['TEST_PATH']}")
        return

    test_data = []
    with open(CONFIG['TEST_PATH'], 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"📊 총 {len(test_data)}개 항목 평가 시작...")
    results = []
    
    for idx, item in enumerate(tqdm(test_data)):
        question = item['question']
        ground_truth = item['answer']
        
        try:
            # A. 실행
            output = app.invoke({"question": question})
            generated_answer = output["answer"]
            
            # B. 문서 정리
            retrieved_docs_dicts = []
            if "retrieved_docs" in output:
                retrieved_docs_dicts = [{"doc_id": d.metadata['doc_id'], "title": d.metadata['title']} for d in output["retrieved_docs"]]
            
            # C. 평가
            metrics = {}
            if evaluator:
                try:
                    metrics = evaluator.evaluate_single_response(
                        question=question,
                        response=generated_answer,
                        ground_truth=ground_truth,
                        retrieved_docs=retrieved_docs_dicts,
                        documents=doc_map
                    )
                except Exception as eval_e:
                    print(f" 평가 에러: {eval_e}")
            
            # D. 저장
            row = {
                "model_name": SELECTED_MODEL,
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "retrieved_doc_ids": [d['doc_id'] for d in retrieved_docs_dicts],
                "rougeL": metrics.get("rougeL", 0),
                "bert_f1": metrics.get("bert_f1", 0),
                "judge_accuracy": metrics.get("accuracy"),
                "judge_helpfulness": metrics.get("helpfulness"),
                "judge_relevance": metrics.get("relevance"),
                "judge_depth": metrics.get("depth")
            }
            results.append(row)
            
        except Exception as e:
            print(f"\n❌ Error at item {idx}: {e}")
            traceback.print_exc()
            continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv(CONFIG["OUTPUT_CSV"], index=False, encoding='utf-8-sig')
        print(f"\n✅ 저장 완료: {CONFIG['OUTPUT_CSV']}")
        try:
            print(df[["rougeL", "bert_f1", "judge_helpfulness"]].mean())
        except:
            pass
    else:
        print("⚠️ 결과 없음")

if __name__ == "__main__":
    main()