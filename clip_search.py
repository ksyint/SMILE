"""
CLIP 기반 이미지 검색 모듈
- clip-retrieval 서비스를 활용한 텍스트 → 이미지 검색
- 텍스트 + 캐릭터 이미지 결합 검색
- 한국어 → 영어 번역 지원

Merges: gradio_demo_api-main/func1.py, func1_clip/query.py
"""

import os
from typing import Optional

from PIL import Image
from clip_retrieval.clip_client import ClipClient
from googletrans import Translator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CLIP_PORT = 13131
DEFAULT_INDICE_PATH = "index_h14"


def _get_client(
    port: int = DEFAULT_CLIP_PORT,
    indice: str = DEFAULT_INDICE_PATH,
    num_images: int = 10,
) -> ClipClient:
    return ClipClient(
        url=f"http://localhost:{port}/knn-service",
        indice_name=indice,
        num_images=num_images,
        deduplicate=False,
        use_safety_model=False,
        use_violence_detector=False,
    )


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

_translator: Optional[Translator] = None


def translate_kor_to_eng(text_kor: str) -> str:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator.translate(text_kor, src="ko", dest="en").text


# ---------------------------------------------------------------------------
# Result Parsing
# ---------------------------------------------------------------------------

def _parse_output(output: dict) -> dict:
    fn1, fn2 = output["image_path"].split("/")[-2:]
    num, _ = os.path.splitext(fn2)
    return {
        "image_path": os.path.join(fn1, fn2),
        "episode": int(fn1),
        "num": int(num),
        "similarity": output["similarity"],
    }


# ---------------------------------------------------------------------------
# Search Functions
# ---------------------------------------------------------------------------

def search_by_text(
    input_text_kor: str,
    input_count: int = 10,
    port: int = DEFAULT_CLIP_PORT,
    indice: str = DEFAULT_INDICE_PATH,
) -> tuple[str, list[dict]]:
    """
    한국어 텍스트를 영어로 번역한 뒤 CLIP 검색을 수행한다.

    Returns:
        (영어 번역 텍스트, 검색 결과 리스트)
    """
    text_en = translate_kor_to_eng(input_text_kor)
    client = _get_client(port, indice, num_images=input_count)
    outputs = client.query(text=text_en)
    results = [_parse_output(o) for o in outputs]
    return text_en, results


def search_by_text_and_image(
    input_text_kor: str,
    character_name: str,
    count_text: int = 1000,
    count_image: int = 1000,
    count_final: int = 10,
    character_image_dir: str = "./character_images",
    port: int = DEFAULT_CLIP_PORT,
    indice: str = DEFAULT_INDICE_PATH,
) -> tuple[str, Image.Image, list[dict]]:
    """
    텍스트 + 캐릭터 이미지 결합 검색.
    텍스트 검색 결과와 이미지 검색 결과를 교차시켜 최종 순위를 결정한다.

    Returns:
        (영어 번역 텍스트, 캐릭터 이미지, 검색 결과 리스트)
    """
    text_en = translate_kor_to_eng(input_text_kor)

    # 텍스트 검색
    client_text = _get_client(port, indice, num_images=count_text)
    outputs_text = client_text.query(text=text_en)

    # 이미지 검색
    img_path = os.path.join(character_image_dir, f"{character_name}.jpg")
    client_img = _get_client(port, indice, num_images=count_image)
    outputs_img = client_img.query(image=img_path)

    # 교차 점수 계산
    combine: dict[str, list[float]] = {}
    for o in outputs_text:
        combine[o["image_path"]] = [o["similarity"]]
    for o in outputs_img:
        combine.setdefault(o["image_path"], []).append(o["similarity"])

    # 두 검색 결과 모두에 포함된 이미지만 선택
    combine_scores = {
        path: vals[0] * vals[1]
        for path, vals in combine.items()
        if len(vals) == 2
    }

    sorted_results = sorted(combine_scores.items(), key=lambda x: x[1],
                            reverse=True)[:count_final]

    results = []
    for path, sim in sorted_results:
        fn1, fn2 = path.split("/")[-2:]
        num, _ = os.path.splitext(fn2)
        results.append({
            "image_path": os.path.join(fn1, fn2),
            "episode": int(fn1),
            "num": int(num),
            "similarity": sim,
        })

    return text_en, Image.open(img_path).convert("RGB"), results
