"""
웹툰 크롤링 모듈
- 네이버 웹툰 에피소드 이미지를 다운로드한다.
- 단일 스레드 / 멀티 스레드 모드를 모두 지원한다.

Merges: crawling-main/crawl_naverWebtoon.py, crawl_naverWebtoon_mp.py
"""

import os
import io
import argparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/83.0.4103.61 Safari/537.36"
    )
}


def fetch_url(url: str) -> requests.Response:
    return requests.get(url, headers=HEADERS)


def parse_and_save_images(html: requests.Response, save_dir: str) -> None:
    html_parse = BeautifulSoup(html.content, "html.parser")
    viewer = html_parse.find("div", {"class": "wt_viewer"})
    if viewer is None:
        print(f"[WARN] viewer not found for {save_dir}")
        return
    webtoon_images = viewer.findAll("img")
    for webtoon_image in tqdm(webtoon_images, desc=save_dir, leave=False):
        src = webtoon_image.get("src", "")
        if not src:
            continue
        base = os.path.basename(src)
        fn, _ = os.path.splitext(base)
        fn_key = fn.split("_")[-1]
        save_path = os.path.join(save_dir, f"{fn_key.zfill(3)}.jpg")

        img_byte = requests.get(src, headers=HEADERS)
        try:
            Image.open(io.BytesIO(img_byte.content)).save(save_path)
        except Exception as e:
            print(f"Error: {e} / {src}")


def crawl_single_thread(toon_id: int, epi_start: int, epi_end: int,
                         webtoon_name: str) -> None:
    for epi in tqdm(range(epi_start, epi_end + 1), desc="Episodes"):
        page_url = (
            f"https://comic.naver.com/webtoon/detail"
            f"?titleId={toon_id}&no={epi}"
        )
        save_dir = os.path.join(webtoon_name, str(epi).zfill(4))
        os.makedirs(save_dir, exist_ok=True)

        html = fetch_url(page_url)
        parse_and_save_images(html, save_dir)


def crawl_multi_thread(toon_id: int, epi_start: int, epi_end: int,
                        webtoon_name: str, max_workers: int = 128) -> None:
    urls, save_dirs = [], []
    for epi in range(epi_start, epi_end + 1):
        page_url = (
            f"https://comic.naver.com/webtoon/detail"
            f"?titleId={toon_id}&no={epi}"
        )
        save_dir = os.path.join(webtoon_name, str(epi).zfill(4))
        os.makedirs(save_dir, exist_ok=True)
        urls.append(page_url)
        save_dirs.append(save_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        htmls = list(executor.map(fetch_url, urls))
        for idx, html in enumerate(htmls):
            parse_and_save_images(html, save_dirs[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="네이버 웹툰 크롤러")
    parser.add_argument("--toon_id", type=int, default=743838)
    parser.add_argument("--epi_start", type=int, default=1)
    parser.add_argument("--epi_end", type=int, default=195)
    parser.add_argument("--webtoon_name", type=str, default="소녀재판")
    parser.add_argument("--max_workers", type=int, default=128,
                        help="0 이면 단일 스레드, 1 이상이면 멀티 스레드")
    args = parser.parse_args()

    if args.max_workers <= 0:
        crawl_single_thread(args.toon_id, args.epi_start, args.epi_end,
                             args.webtoon_name)
    else:
        crawl_multi_thread(args.toon_id, args.epi_start, args.epi_end,
                            args.webtoon_name, args.max_workers)
