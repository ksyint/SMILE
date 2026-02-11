"""
OCR 및 마스크 유틸리티 모듈
- EasyOCR 기반 웹툰 이미지에서 텍스트(대사) 추출
- 말풍선 영역 마스킹 및 바운딩 박스 추출
- 텍스트 정제 및 유사도 계산

Merges: dialogue_search-main/web_ocr.py, mask.py, sim.py
"""

import os
import re
import glob
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image
import easyocr
import Levenshtein


# ---------------------------------------------------------------------------
# Mask / Bounding Box
# ---------------------------------------------------------------------------

def create_image_mask(
    coordinates_list: list,
    image_shape: tuple,
    save_path: str = "mask.png",
) -> Image.Image:
    """OCR 좌표 목록으로부터 마스크 이미지를 생성한다."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    for coords, _, _ in coordinates_list:
        pts = np.array(coords, dtype=np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
    mask_img = Image.fromarray(mask)
    mask_img.save(save_path)
    return mask_img


def extract_bounding_boxes(mask_path: str) -> list[tuple[int, int, int, int]]:
    """마스크 이미지에서 흰 영역의 바운딩 박스를 추출한다."""
    image = cv2.imread(mask_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    return boxes


def crop_speech_bubbles(
    mask_path: str,
    original_image_path: str,
    output_dir: str,
) -> list[str]:
    """마스크를 기준으로 원본 이미지에서 말풍선 영역을 잘라낸다."""
    os.makedirs(output_dir, exist_ok=True)
    image = np.array(Image.open(original_image_path))[:, :, :3]
    boxes = extract_bounding_boxes(mask_path)

    saved_paths = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        crop = image[y1:y2, x1:x2, :]
        out_path = os.path.join(output_dir, f"box_{idx + 1}.png")
        Image.fromarray(crop).save(out_path)
        saved_paths.append(out_path)
    return saved_paths


# ---------------------------------------------------------------------------
# Text Utils
# ---------------------------------------------------------------------------

def clean_and_combine_text(text_list: list[str]) -> str:
    """OCR 결과 텍스트 리스트를 정제하고 하나로 합친다."""
    cleaned = [re.sub(r"[^a-zA-Z가-힣0-9\s]", "", t).strip() for t in text_list]
    combined = " ".join(cleaned)
    combined = re.sub(r"\s+", " ", combined)
    return combined


def calculate_similarity(sentence1: str, sentence2: str) -> float:
    """레벤슈타인 거리 기반 유사도를 반환한다 (0~1)."""
    distance = Levenshtein.distance(sentence1, sentence2)
    max_len = max(len(sentence1), len(sentence2))
    if max_len == 0:
        return 1.0
    return 1 - (distance / max_len)


def extract_path_after_keyword(file_path: str, keyword: str) -> Optional[str]:
    idx = file_path.find(keyword)
    return file_path[idx:] if idx != -1 else None


# ---------------------------------------------------------------------------
# Batch OCR Pipeline
# ---------------------------------------------------------------------------

def batch_ocr_pipeline(
    image_dir: str,
    episodes: Optional[list[str]] = None,
    box_dir: str = "./box",
    output_prefix: str = "webtoon_list",
    save_every: int = 100,
) -> list[dict]:
    """
    이미지 디렉터리 내 에피소드 이미지들에 대해 OCR을 수행한다.

    Args:
        image_dir: 이미지 상위 디렉터리 (에피소드별 하위 폴더 포함)
        episodes: 처리 대상 에피소드 ID 리스트 (None이면 전체)
        box_dir: 말풍선 crop 임시 저장 경로
        output_prefix: 결과 JSON 파일 이름 접두어
        save_every: 중간 저장 주기 (N개마다)

    Returns:
        OCR 결과 리스트 (각 이미지의 경로 + 대사 목록)
    """
    reader = easyocr.Reader(["ko", "en"])
    paths = sorted(glob.glob(os.path.join(image_dir, "*/*.jpg")))
    main_list: list[dict] = []
    count = 0

    for idx, path in enumerate(paths):
        episode = path.split(os.sep)[-2]
        if episodes is not None and episode not in episodes:
            continue

        count += 1
        img = np.array(Image.open(path))[:, :, :3]
        result = reader.readtext(path)

        # 마스크 생성 → 말풍선 crop
        mask_path = "mask.png"
        create_image_mask(result, img.shape, save_path=mask_path)
        bubble_paths = crop_speech_bubbles(mask_path, path, box_dir)
        if os.path.exists(mask_path):
            os.remove(mask_path)

        # 각 말풍선에서 텍스트 추출
        text_list = []
        for bp in bubble_paths:
            texts = reader.readtext(bp, detail=0)
            text_list.append(clean_and_combine_text(texts))

        entry = {
            "path": extract_path_after_keyword(path, "orig-result") or path,
            "sentences": text_list,
        }
        main_list.append(entry)

        # 임시 폴더 정리
        shutil.rmtree(box_dir, ignore_errors=True)
        os.makedirs(box_dir, exist_ok=True)

        # 중간 저장
        if count > 0 and count % save_every == 0:
            out_file = f"{output_prefix}_{idx}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(main_list, f, ensure_ascii=False, indent=4)
            main_list = []
            print(f"[OCR] 중간 저장 완료: {out_file}")

    return main_list


# ---------------------------------------------------------------------------
# Simple Dialogue Search
# ---------------------------------------------------------------------------

def find_most_similar_desc(
    query: str,
    data: dict,
) -> Optional[str]:
    """
    대사 데이터에서 query와 가장 유사한 설명(desc)이 있는 이미지 경로를 반환한다.
    """
    max_sim = -1.0
    best_img_path = None

    for doc in data.get("all_doc", []):
        for dialogue in doc.get("dialogue", []):
            desc = dialogue.get("desc", "")
            sim = calculate_similarity(query, desc)
            if sim > max_sim:
                max_sim = sim
                best_img_path = doc.get("img_path")

    return best_img_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR 배치 파이프라인")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--box_dir", type=str, default="./box")
    parser.add_argument("--output_prefix", type=str, default="webtoon_list")
    args = parser.parse_args()

    results = batch_ocr_pipeline(
        args.image_dir,
        episodes=args.episodes,
        box_dir=args.box_dir,
        output_prefix=args.output_prefix,
    )
    if results:
        out_file = f"{args.output_prefix}_final.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"최종 저장: {out_file}")
