import os
import json
import openai
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
with open("secret.txt", 'r') as f :
    openai.api_key = f.readline().strip()
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
    data_path = f"./func2_retrieval/data/{data_type}_data.json"
    db_name = f"./func2_retrieval/faiss_index_{data_type}"
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


def search_by_dialogue(query, count):
    db = load_db(data_type="dialogue")
    results_with_scores = db.similarity_search_with_score(query, k=count)
    
    search_result = []
    for doc, score in results_with_scores:
        episode, num = doc.metadata['webtoon_num'], doc.metadata['cut_num']
        image_path = f"{str(episode).zfill(4)}/{str(num).zfill(3)}.jpg"
        temp = {'content': doc.page_content,
                'image_path': image_path,
                'episode': episode,
                'num': num,
                'similarity': score}
        search_result.append(temp)    
        
    return search_result

def refine_query(query):
    prompt = f"""질문에서 아래 조건에 따라 '장면', '대사', '등장인물'을 key 값으로 가지는 JSON 데이터를 반환해줘.

    1. 장면 추출: 질문에서 추출된 장면을 영어로 작성해줘. 등장인물의 이름은 이름 대신 등장인물의 성별로 대체해서 작성해줘.
    - 한유현, 강효민, 차태석: A boy
    - 박가을: A girl
    2. 대사 추출: 만약 질문에서 웹툰 대사가 언급됐다면, 등장인물의 실제 대사처럼 재구성해서 출력하고, 없으면 None을 반환해줘.
    3. 주인공 이름 추출:  만약 질문에서 등장인물의 이름이 등장한 경우, 등장인물의 이름을 리스트 형태로 반환하고, 없으면 None을 반환해줘. 이름은 반드시 3글자로 반환해줘.

    질문: {query}"""
    
    response = openai.ChatCompletion.create(
        max_tokens=256,
        temperature=0.0,
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