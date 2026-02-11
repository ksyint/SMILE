"""
웹툰 검색 데모 애플리케이션
- Gradio UI + FastAPI 통합
- CLIP 텍스트 검색, 대사 검색, Composed Image 검색, 통합 검색

Merges: gradio_demo_api-main/demo.py
"""

import os
import argparse
from typing import Optional

import gradio as gr
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from clip_search import search_by_text, search_by_text_and_image
from text_search import search_by_dialogue, refine_query
from composed_search import do_retrieve, load_character_embeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_DIR = "./dataset/orig-result"
CHARACTERS = ["강효민", "박가을", "차태석", "한유현", "황윤혜"]


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="웹툰 검색 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextItem(BaseModel):
    text_kor: str


@app.post("/search-by-text",
          summary="(clip-retrieval) 텍스트(장면)으로 이미지 검색")
def api_search_by_text(req_json: TextItem):
    text_en, search_list = search_by_text(req_json.text_kor, input_count=5)
    return {"text_en": text_en, "search_list": search_list}


@app.post("/search-by-dialogue",
          summary="(LaBSE) 텍스트(대사)로 이미지 검색")
def api_search_by_dialogue(req_json: TextItem):
    search_list = search_by_dialogue(req_json.text_kor, count=2)
    return {"search_list": search_list}


@app.post("/search-by-final",
          summary="텍스트(장면 & 대사) 통합 검색")
def api_search_by_final(req_json: TextItem):
    middle_text, _, search_list = search_by_final(req_json.text_kor)
    return {"middle_text": middle_text, "search_list": search_list}


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def search_by_final(input_text_kor: str):
    """GPT 분석 → 대사/장면/캐릭터 조건에 따라 검색 경로를 분기한다."""
    query_info = refine_query(input_text_kor)
    scene = query_info.get("장면", "")
    dialogue = query_info.get("대사")
    character = query_info.get("등장인물")

    if dialogue in ("None", "none", "Null", "null", None):
        dialogue = None

    if dialogue:
        search_list = search_by_dialogue(query=dialogue, count=2)
        return f"{query_info}\n[search_by_dialogue]", None, search_list

    # 캐릭터 결합 검색 또는 텍스트 단독 검색
    if isinstance(character, list):
        target_character = None
        for c in CHARACTERS:
            if c in character:
                target_character = c
                break

        if target_character:
            text_en, char_img, search_list = search_by_text_and_image(
                scene, target_character, 500, 500, 5
            )
        else:
            text_en, search_list = search_by_text(scene, input_count=5)
            char_img = None
    else:
        text_en, search_list = search_by_text(scene, input_count=5)
        char_img = None

    return (
        f"{query_info}\n[search_by_text]\n{text_en}",
        char_img,
        search_list,
    )


def parsing_json_for_display(search_list: list[dict]) -> list[Image.Image]:
    """검색 결과 JSON을 이미지 리스트로 변환한다."""
    outputs = []
    for item in search_list:
        img_path = os.path.join(IMAGE_DIR, item["image_path"])
        if os.path.exists(img_path):
            outputs.append(Image.open(img_path).convert("RGB"))
    return outputs


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

def build_gradio_app() -> gr.Blocks:
    with gr.Blocks() as demo:

        # --- Tab 1: CLIP Text Search ---
        with gr.Tab("text search (clip-retrieval)"):
            with gr.Row():
                with gr.Column():
                    t1_text = gr.Text(label="Input (Kor)",
                                      value="한 학생이 울고 있는 장면")
                    t1_count = gr.Slider(label="Max count", minimum=1,
                                         maximum=100, step=1, value=10)
                    t1_btn = gr.Button("Submit", variant="primary")
                with gr.Column():
                    t1_en = gr.Text(label="Input (En)", interactive=False)
                    t1_json = gr.Json(label="Results")
                    t1_gallery = gr.Gallery(label="Output images", columns=5)

            t1_btn.click(
                fn=search_by_text,
                inputs=[t1_text, t1_count],
                outputs=[t1_en, t1_json],
                concurrency_id="default",
                api_name="search_by_text",
            ).then(
                fn=parsing_json_for_display,
                inputs=[t1_json],
                outputs=[t1_gallery],
                concurrency_id="default",
                show_api=False,
            )

        # --- Tab 2: Dialogue Search ---
        with gr.Tab("dialogue search (LaBSE)"):
            with gr.Row():
                with gr.Column():
                    t2_text = gr.Text(label="Input (Kor)",
                                      value="안경 낀 사람 때리면 살인")
                    t2_count = gr.Slider(label="Max count", minimum=1,
                                         maximum=100, step=1, value=2)
                    t2_btn = gr.Button("Submit", variant="primary")
                with gr.Column():
                    t2_json = gr.Json(label="Results")
                    t2_gallery = gr.Gallery(label="Output images", columns=5)

            t2_btn.click(
                fn=search_by_dialogue,
                inputs=[t2_text, t2_count],
                outputs=[t2_json],
                concurrency_id="default",
                api_name="search_by_dialogue",
            ).then(
                fn=parsing_json_for_display,
                inputs=[t2_json],
                outputs=[t2_gallery],
                concurrency_id="default",
                show_api=False,
            )

        # --- Tab 3: Final (Unified) Search ---
        with gr.Tab("final search"):
            with gr.Row():
                with gr.Column():
                    tf_text = gr.Text(label="Input (Kor)",
                                      value="안경 낀 사람 때리면 살인")
                    tf_btn = gr.Button("Submit", variant="primary")
                with gr.Column():
                    with gr.Row():
                        tf_mid = gr.Text(label="Middle text", scale=2)
                        tf_img = gr.Image(label="캐릭터 이미지", scale=1,
                                          interactive=False)
                    tf_json = gr.Json(label="Results")
                    tf_gallery = gr.Gallery(label="Output images", columns=5)

            tf_btn.click(
                fn=search_by_final,
                inputs=[tf_text],
                outputs=[tf_mid, tf_img, tf_json],
                concurrency_id="default",
            ).then(
                fn=parsing_json_for_display,
                inputs=[tf_json],
                outputs=[tf_gallery],
                concurrency_id="default",
                show_api=False,
            )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="웹툰 검색 데모")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR)
    args = parser.parse_args()

    IMAGE_DIR = args.image_dir

    demo = build_gradio_app()
    demo.queue()
    gr_app = gr.mount_gradio_app(app, demo, path="/demo")
    uvicorn.run(app, host=args.host, port=args.port)
