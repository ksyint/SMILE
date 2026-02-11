"""
텍스트(대사/장면) 검색 모듈
- FAISS + LaBSE 임베딩 기반 장면/대사 유사도 검색
- OpenAI GPT 기반 쿼리 분석 (장면, 대사, 등장인물 추출)

Merges: gradio_demo_api-main/func2.py, text_retrieval_merged.py
"""

import os
import json
from typing import Optional

import openai
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Embeddings (모듈 로드 시 초기화)
# ---------------------------------------------------------------------------

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/LaBSE"
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def read_json_file(file_path: str) -> list | dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scene_data(data_path: str) -> list[Document]:
    data = read_json_file(data_path)
    documents = []
    for datum in data:
        webtoon_num = int(datum["img_path"].split("/")[1])
        cut_num = int(datum["img_path"].split("/")[2].replace(".jpg", ""))
        desc = datum["desc"]
        if "###" in desc:
            desc = desc.split("\n###")[0].strip()
        documents.append(
            Document(
                page_content=desc,
                metadata={"webtoon_num": webtoon_num, "cut_num": cut_num},
            )
        )
    return documents


def load_dialogue_data(data_path: str) -> list[Document]:
    data = read_json_file(data_path)
    documents = []
    for datum in data:
        webtoon_num = int(datum["img_path"].split("/")[1])
        cut_num = int(datum["img_path"].split("/")[2].replace(".jpg", ""))
        for dialogue in datum.get("dialogue", []):
            dialogue_num = dialogue["dialogue_num"]
            desc = dialogue["desc"]
            documents.append(
                Document(
                    page_content=desc,
                    metadata={
                        "webtoon_num": webtoon_num,
                        "cut_num": cut_num,
                        "dialogue_num": dialogue_num,
                    },
                )
            )
    return documents


# ---------------------------------------------------------------------------
# FAISS DB
# ---------------------------------------------------------------------------

def load_db(
    data_type: str,
    data_dir: str = "./data",
    db_dir: str = ".",
) -> FAISS:
    """
    FAISS 인덱스를 로드하거나 새로 생성한다.

    Args:
        data_type: "scene" 또는 "dialogue"
        data_dir: JSON 데이터 경로
        db_dir: FAISS 인덱스 저장 경로
    """
    data_path = os.path.join(data_dir, f"{data_type}_data.json")
    db_name = os.path.join(db_dir, f"faiss_index_{data_type}")
    embeddings = get_embeddings()

    if not os.path.exists(db_name):
        if data_type == "scene":
            docs = load_scene_data(data_path)
        elif data_type == "dialogue":
            docs = load_dialogue_data(data_path)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(db_name)
    else:
        print(f"Load pre-saved FAISS index: {db_name}")
        db = FAISS.load_local(db_name, embeddings,
                              allow_dangerous_deserialization=True)
    return db


# ---------------------------------------------------------------------------
# Search Functions
# ---------------------------------------------------------------------------

def search_by_dialogue(
    query: str,
    count: int = 5,
    data_dir: str = "./data",
    db_dir: str = ".",
) -> list[dict]:
    """대사 기반 유사도 검색"""
    db = load_db("dialogue", data_dir=data_dir, db_dir=db_dir)
    results_with_scores = db.similarity_search_with_score(query, k=count)

    search_result = []
    for doc, score in results_with_scores:
        episode = doc.metadata["webtoon_num"]
        num = doc.metadata["cut_num"]
        image_path = f"{str(episode).zfill(4)}/{str(num).zfill(3)}.jpg"
        search_result.append({
            "content": doc.page_content,
            "image_path": image_path,
            "episode": episode,
            "num": num,
            "similarity": float(score),
        })
    return search_result


def search_by_scene(
    query: str,
    count: int = 5,
    data_dir: str = "./data",
    db_dir: str = ".",
) -> list[dict]:
    """장면 기반 유사도 검색"""
    db = load_db("scene", data_dir=data_dir, db_dir=db_dir)
    results_with_scores = db.similarity_search_with_score(query, k=count)

    search_result = []
    for doc, score in results_with_scores:
        episode = doc.metadata["webtoon_num"]
        num = doc.metadata["cut_num"]
        image_path = f"{str(episode).zfill(4)}/{str(num).zfill(3)}.jpg"
        search_result.append({
            "content": doc.page_content,
            "image_path": image_path,
            "episode": episode,
            "num": num,
            "similarity": float(score),
        })
    return search_result


# ---------------------------------------------------------------------------
# Query Refinement (GPT)
# ---------------------------------------------------------------------------

def refine_query(
    query: str,
    model: str = "gpt-4-1106-preview",
) -> dict:
    """
    GPT를 사용하여 사용자 쿼리를 분석한다.
    반환 키: 장면, 대사, 등장인물
    """
    prompt = f"""질문에서 아래 조건에 따라 '장면', '대사', '등장인물'을 key 값으로 가지는 JSON 데이터를 반환해줘.

    1. 장면 추출: 질문에서 추출된 장면을 영어로 작성해줘. 등장인물의 이름은 이름 대신 등장인물의 성별로 대체해서 작성해줘.
    - 한유현, 강효민, 차태석: A boy
    - 박가을: A girl
    2. 대사 추출: 만약 질문에서 웹툰 대사가 언급됐다면, 등장인물의 실제 대사처럼 재구성해서 출력하고, 없으면 None을 반환해줘.
    3. 주인공 이름 추출: 만약 질문에서 등장인물의 이름이 등장한 경우, 등장인물의 이름을 리스트 형태로 반환하고, 없으면 None을 반환해줘. 이름은 반드시 3글자로 반환해줘.

    질문: {query}"""

    response = openai.ChatCompletion.create(
        max_tokens=256,
        temperature=0.0,
        response_format={"type": "json_object"},
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant who responds in json."},
            {"role": "user", "content": prompt},
        ],
    )

    result = response.choices[0].message.content
    return json.loads(result)
