import os
import json
from PIL import Image
import gradio as gr
from gradio_client import Client
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import argparse

from func1 import search_by_text, search_by_text_and_image
from func2 import search_by_dialogue, refine_query
from func3 import get_target_paths, do_retrieve

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=7000)
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()

IMAGE_DIR = "../dataset_shared2/orig-result"
CHARACTERS = ["강효민", "박가을", "차태석", "한유현", "황윤혜"]

app = FastAPI()

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class TextItem(BaseModel):
    text_kor: str

@app.post('/search-by-text', summary="(clip-retrieval) 텍스트(장면)으로 이미지 검색",
          description="장면에 대한 한국어 문장을 입력하면 이미지를 검색합니다.")
def api_search_by_text(req_json: TextItem) :
    text_kor = req_json.text_kor
    count = 5
    
    client = Client(f"http://localhost:{args.port}/demo")
    text_en, search_list = client.predict(api_name="/search_by_text", 
                                          input_text_kor=text_kor, input_count=count)
    return {"text_en" : text_en,
            "search_list" : search_list}

@app.post('/search-by-dialogue', summary="(chatgpt + LaBSE) 택스트(대사)로 이미지 검색",
          description="대사에 대한 한국어 문장을 입력하면 이미지를 검색합니다.")
def api_search_by_dialogue(req_json: TextItem) :
    text_kor = req_json.text_kor
    count = 2

    client = Client(f"http://localhost:{args.port}/demo")
    search_list = client.predict(api_name="/search_by_dialogue", 
                                 query=text_kor, count=count)
    return {"search_list" : search_list}


@app.post('/search-by-final', summary="텍스트(장면 & 대사)로 이미지 검색",
          description="한국어 문장을 입력하면 이미지를 검색합니다.")
def api_search_by_final(req_json: TextItem) :
    text_kor = req_json.text_kor

    client = Client(f"http://localhost:{args.port}/demo")
    middle_text, _, search_list = client.predict(api_name="/search_by_final", input_text_kor=text_kor)
    return {"middle_text": middle_text,
            "search_list" : search_list}

def search_by_final(input_text_kor) :
    query_info_dict = refine_query(input_text_kor)
    scene = query_info_dict["장면"]
    dialogue = query_info_dict["대사"]
    character = query_info_dict["등장인물"]

    if dialogue in ['None', 'none', 'Null', 'null'] : dialogue = None

    if dialogue :        
        search_list = search_by_dialogue(query=dialogue, count=2)
        return f'{query_info_dict}\n[search_by_dialogue]', None, search_list
    else :
        if type(character) == list :
            target_character = None
            for CHARACTER in CHARACTERS :
                if CHARACTER in character :
                    target_character = CHARACTER
                    break

            if target_character == None :
                text_en, search_list = search_by_text(scene, input_count=5)
                char_img = None
            else :
                text_en, char_img, search_list = search_by_text_and_image(scene, target_character, 500, 500, 5)
        else :
            text_en, search_list = search_by_text(scene, input_count=5)
            char_img = None
            
        return f'{query_info_dict}\n[search_by_text]\n{text_en}', char_img, search_list

def parsing_json_for_display(search_list) :
    outputs = []
    for search in search_list :
        img_path = os.path.join(IMAGE_DIR, search['image_path'])
        img = Image.open(img_path).convert('RGB')
        outputs.append(img)
    return outputs

with gr.Blocks() as demo :

    with gr.Tab("text search (clip-retrieval)") :
        with gr.Row() :
            with gr.Column() :
                func1_input_text_kor = gr.Text(label="Input (Kor)", info="한국어로 질문을 입력하세요", value="한 학생이 울고 있는 장면")
                func1_input_count = gr.Slider(label="Max count", info="응답받는 최대 개수를 설정합니다.", minimum=1, maximum=100, step=1, value=10)
                func1_btn_submit = gr.Button(value="Submit", variant='primary')

            with gr.Column() :
                func1_output_text_en = gr.Text(label="Input (En)", info="한국어를 영어로 번역", interactive=False)
                func1_output_list = gr.Json(label="Outpus")
                func1_output_gallery = gr.Gallery(label="Output images", columns=5)

        func1_btn_submit.click(fn=search_by_text,
                               inputs=[func1_input_text_kor, func1_input_count],
                               outputs=[func1_output_text_en, func1_output_list],
                               concurrency_id='default',
                               api_name='search_by_text').then(fn=parsing_json_for_display,
                                                              inputs=[func1_output_list],
                                                              outputs=[func1_output_gallery],
                                                              concurrency_id='default',
                                                              show_api=False)

    with gr.Tab("dialogue search (chatgpt + LaBSE)") :
        with gr.Row() :
            with gr.Column() :
                func2_input_text_kor = gr.Text(label="Input (Kor)", info="한국어로 질문을 입력하세요", value="안경 낀 사람 때리면 살인 대사 몇화인지 알려주세요")
                func2_input_count = gr.Slider(label="Max count", info="응답받는 최대 개수를 설정합니다.", minimum=1, maximum=100, step=1, value=2)
                func2_btn_submit = gr.Button(value="Submit", variant='primary')

            with gr.Column() :
                func2_output_list = gr.Json(label="Outpus")
                func2_output_gallery = gr.Gallery(label="Output images", columns=5)

        func2_btn_submit.click(fn=search_by_dialogue,
                               inputs=[func2_input_text_kor, func2_input_count],
                               outputs=[func2_output_list],
                               concurrency_id='default',
                               api_name='search_by_dialogue').then(fn=parsing_json_for_display,
                                                                      inputs=[func2_output_list],
                                                                      outputs=[func2_output_gallery],
                                                                      concurrency_id='default',
                                                                      show_api=False)
        
    with gr.Tab("text with image search (pic2word)") :
        with gr.Row() :
            with gr.Column() :
                func3_input_text_kor = gr.Text(label="Input (Kor)", info="한국어로 질문을 입력하세요. 캐릭터명) 강효민, 김민정, 김서아, 민아름, 박가을, 이미소, 정해서, 진은설, 차태석, 한유현, 황윤혜",
                                               value="there exists 박가을")
                func3_input_count = gr.Slider(label="Max count", info="응답받는 최대 개수를 설정합니다.", minimum=1, maximum=100, step=1, value=5)
                func3_input_paths = gr.Textbox(label="Target paths", info="대상으로 하는 이미지 경로", lines=10, value=get_target_paths('sample'))
                func3_btn_submit = gr.Button(value="Submit", variant='primary')

            with gr.Column() :
                func3_output_text_en = gr.Text(label="Input (En)", info="한국어를 영어로 번역", interactive=False)
                func3_output_img = gr.Image(label='Character image', interactive=False)
                func3_output_list = gr.Json(label="Outpus")
                func3_output_gallery = gr.Gallery(label="Output images", columns=5, interactive=False)

        func3_btn_submit.click(fn=do_retrieve,
                               inputs=[func3_input_text_kor, func3_input_paths, func3_input_count],
                               outputs=[func3_output_text_en, func3_output_img, func3_output_list],
                               concurrency_id='default').then(fn=parsing_json_for_display,
                                                              inputs=[func3_output_list],
                                                              outputs=[func3_output_gallery],
                                                              concurrency_id='default',
                                                              show_api=False)
        
    with gr.Tab("final search") :
        with gr.Row() :
            with gr.Column() :
                final_input_text_kor = gr.Text(label="Input (Kor)", info="한국어로 질문을 입력하세요", value="안경 낀 사람 때리면 살인 대사 몇화인지 알려주세요")
                final_btn_submit = gr.Button(value="Submit", variant='primary')

            with gr.Column() :
                with gr.Row() :
                    middle_text = gr.Text(label="Middle text", info="중간 과정의 텍스트", scale=2)
                    final_output_img = gr.Image(label="캐릭터 이미지", scale=1, interactive=False)
                final_output_list = gr.Json(label="Outpus")
                final_output_gallery = gr.Gallery(label="Output images", columns=5)

        final_btn_submit.click(fn=search_by_final,
                               inputs=[final_input_text_kor],
                               outputs=[middle_text, final_output_img, final_output_list],
                               concurrency_id='default',
                               api_name='search_by_final').then(fn=parsing_json_for_display,
                                                              inputs=[final_output_list],
                                                              outputs=[final_output_gallery],
                                                              concurrency_id='default',
                                                              show_api=False)

demo.title = "웹툰검색데모"
demo.queue(default_concurrency_limit=1)
# demo.launch(server_name='0.0.0.0', server_port=7000, share=False)

app = gr.mount_gradio_app(app, demo, path='/demo')
uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)