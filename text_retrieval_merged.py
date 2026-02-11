import os
import json
from fastapi import FastAPI
from fastapi.responses import Response
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import openai
openai.api_key = ""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def load_scene_data(data_path):
    list_of_documents = []
    data = read_json_file(data_path)
    for datum in data:
        webtoon_num = int(datum['img_path'].split('/')[1])
        cut_num = int(datum['img_path'].split('/')[2].replace('.jpg', ''))
        desc = datum['desc']
        if "###" in desc:
            desc = desc.split("\n###")[0].strip()
        document = Document(page_content=desc, metadata=dict(webtoon_num=webtoon_num, cut_num=cut_num))
        list_of_documents.append(document)
    return list_of_documents


def load_dialogue_data(data_path):
    list_of_documents = []
    data = read_json_file(data_path)
    for datum in data:
        webtoon_num = int(datum['img_path'].split('/')[1])
        cut_num = int(datum['img_path'].split('/')[2].replace('.jpg', ''))
        dialogues = datum['dialogue']
        for dialogue in dialogues:
            dialogue_num = dialogue['dialogue_num']
            desc = dialogue['desc']
            document = Document(page_content=desc, metadata=dict(webtoon_num=webtoon_num, cut_num=cut_num, dialogue_num=dialogue_num))
            list_of_documents.append(document)
    return list_of_documents


def load_db(data_type):
    data_path = f"./data/{data_type}_data.json"
    db_name = f"faiss_index_{data_type}"
    if not os.path.exists(db_name):
        if data_type == "scene":
            list_of_documents = load_scene_data(data_path)
        if data_type == "dialogue":
            list_of_documents = load_dialogue_data(data_path)
        db = FAISS.from_documents(list_of_documents, embeddings)
        db.save_local(db_name)
    else:
        print("Load pre-saved FAISS index")
        db = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
    return db


def refine_query(query):
    prompt = f"""질문에서 아래 조건에 따라 '장면', '대사', '등장인물'을 key 값으로 가지는 JSON 데이터를 반환해줘.

    1. 장면 추출: 등장인물의 이름 대신 등장인물의 성별로 대체해서 작성해줘.
    - 한유현, 강효민, 차태석: 남성
    - 박가을: 여성
    2. 대사 추출: 만약 질문에서 웹툰 대사가 언급됐다면, 등장인물의 실제 대사처럼 재구성해서 출력하고, 없으면 None을 반환해줘.
    3. 주인공 이름 추출:  만약 질문에서 등장인물의 이름이 등장한 경우, 등장인물의 이름을 리스트 형태로 반환하고, 없으면 None을 반환해줘. 이름은 반드시 3글자로 반환해줘.

    질문: {query}"""
    
    response = openai.ChatCompletion.create(
        max_tokens=256,
        response_format = {"type": "json_object"},
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who responds in json."},
            {"role": "user", "content": prompt},
        ]
    )

    result = response.choices[0].message.content
    result_dict = json.loads(result)
    return result_dict


def search(query, search_type):
    db = load_db(data_type=search_type)
    results = db.similarity_search_with_score(query)
    search_result = []
    for doc, score in results:
        temp = {'content': doc.page_content,
                'webtoon_num': doc.metadata['webtoon_num'],
                'cut_num': doc.metadata['cut_num'],
                'score': score}
        search_result.append(temp)
    json_str = json.dumps(search_result, indent=4, default=str, ensure_ascii = False)
    return json_str
    

app = FastAPI()
@app.post("/search")
async def search_query(query: str):
    query_info_dict = refine_query(query)
    scene = query_info_dict["장면"]
    dialogue = query_info_dict["대사"]
    character = query_info_dict["등장인물"]
    print(query_info_dict)
        
    search_by_scene = search(scene, "scene")
    search_by_dialogue = search(dialogue, "dialogue")
    search_by_all = [{"search_type": "scene", "search_result": search_by_scene},
                     {"search_type": "dialogue", "search_result": search_by_dialogue}]

    json_str = json.dumps(search_by_all, indent=4, default=str, ensure_ascii = False)
    return Response(content=json_str, media_type='application/json')
