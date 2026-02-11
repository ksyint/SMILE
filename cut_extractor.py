"""
웹툰 컷 추출 모듈
- 웹툰 이미지 청크들을 행 단위로 스캔하여 개별 컷(장면)으로 분리한다.
- 흰 배경 / 검정 배경 기반 분리를 지원한다.

Merges: cut-extractor-main/main.py, image_loader.py
"""

from pathlib import Path
from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable, Deque, Optional, Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import cv2
import click
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Image Loaders
# ---------------------------------------------------------------------------

class ImageLoaderBase(ABC):
    @abstractmethod
    def iter_rows(self) -> Iterable[NDArray[np.uint8]]:
        pass


class ChunkImageLoader(ImageLoaderBase):
    """디렉터리 내 jpg 이미지 청크들을 로드하고 행 단위로 순회한다."""

    def __init__(self, episode_path: Path | str):
        self.episode_path = Path(episode_path)

    def load_img_chunk(self, path: Path | str) -> NDArray[np.uint8]:
        im: NDArray[np.uint8] = cv2.cvtColor(
            cv2.imread(str(path)), cv2.COLOR_BGR2RGB
        )
        assert im.dtype == np.uint8
        return im

    def iter_chunks(self) -> Iterable[NDArray[np.uint8]]:
        img_chunk_files = list(self.episode_path.glob("*.jpg"))
        img_chunk_files = [p for p in img_chunk_files if p.stem != "white"]
        img_chunk_files.sort(key=lambda p: p.stem)
        for chunk_file in img_chunk_files:
            yield self.load_img_chunk(chunk_file)

    def iter_rows(self) -> Iterable[NDArray[np.uint8]]:
        for img_chunk in self.iter_chunks():
            yield from img_chunk

    def load(self) -> NDArray[np.uint8]:
        return np.concatenate([*self.iter_chunks()], axis=0)


class ArrayImageLoader(ImageLoaderBase):
    """이미 로드된 numpy 배열에서 행 단위로 순회한다."""

    def __init__(self, img_array: NDArray[np.uint8]) -> None:
        self.img_array = img_array

    def iter_rows(self) -> Iterable[NDArray[np.uint8]]:
        yield from self.img_array


# ---------------------------------------------------------------------------
# Pixel Helpers
# ---------------------------------------------------------------------------

def is_all_black(img_arr: NDArray[np.uint8], tol: float = 25.0) -> bool:
    assert img_arr.dtype == np.uint8
    return bool(np.all(img_arr.sum(axis=-1) < tol))


def is_all_white(img_arr: NDArray[np.uint8], tol: float = 25.0) -> bool:
    inverted = (255 - img_arr).astype(np.uint8)
    return bool(np.all(inverted.sum(axis=-1) < tol))


# ---------------------------------------------------------------------------
# Cut Extractor
# ---------------------------------------------------------------------------

class CutExtractor:
    """
    웹툰 이미지에서 컷(장면) 단위로 분리하여 추출한다.
    빈 행(흰색 또는 검정)을 기준으로 이미지를 분할한다.
    """

    def __init__(
        self,
        img_loader: ImageLoaderBase,
        min_height: int = 100,
        max_height: int = 5_000,
        color_tol: float = 50.0,
        black_bg: bool = False,
        ep_dir: Optional[str | Path] = None,
        dst: str | Path = "./result",
    ):
        if ep_dir is None:
            assert isinstance(img_loader, ChunkImageLoader)
            ep_dir = img_loader.episode_path
        self.ep_dir = Path(ep_dir)
        assert self.ep_dir.is_dir()

        self.img_loader = img_loader
        self.min_height = min_height
        self.max_height = max_height
        self.black_bg = black_bg

        self.row_buf: Deque[NDArray[np.uint8]] = deque()
        self.color_tol = color_tol
        self.dst = Path(dst)

    @property
    def white_bg(self) -> bool:
        return not self.black_bg

    # ------ core iteration ------

    def is_empty(self, row: NDArray[np.uint8]) -> bool:
        return (
            is_all_black(row, self.color_tol)
            if self.black_bg
            else is_all_white(row, self.color_tol)
        )

    def get_candid_cut(self) -> NDArray[np.uint8] | None:
        return np.stack(self.row_buf, axis=0) if self.row_buf else None

    def iter_cand_cuts(self) -> Iterable[NDArray[np.uint8]]:
        for row in self.img_loader.iter_rows():
            if not self.is_empty(row):
                self.row_buf.append(row)
                continue
            cut = self.get_candid_cut()
            if cut is not None:
                yield cut

    # ------ extract pipeline ------

    def extract(self) -> Iterable[Image.Image]:
        for cand_cut in self.iter_cand_cuts():
            yield from self._proc_cand_cut(cand_cut)
        yield from self._finalize()

    def extract_and_save(self) -> None:
        out_dir = self.dst / self.ep_dir.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(self.extract()):
            im.save(out_dir / f"{i:03}.jpg")

    def _finalize(self) -> Iterable[Image.Image]:
        cut = self.get_candid_cut()
        if cut is not None:
            yield from self._proc_cand_cut(cut)

    def _proc_cand_cut(self, cut: NDArray[np.uint8]) -> Iterable[Image.Image]:
        H, W, C = cut.shape
        too_high = H > self.max_height

        if too_high and self.white_bg:
            yield from CutExtractor(
                ArrayImageLoader(cut),
                self.min_height,
                self.max_height,
                black_bg=True,
                color_tol=self.color_tol,
                ep_dir=self.ep_dir,
            ).extract()
            self.row_buf.clear()
            return

        if too_high and self.black_bg:
            self._save_as_rejected(cut)
            self.row_buf.clear()
            return

        if H > self.min_height:
            yield Image.fromarray(cut)
        self.row_buf.clear()

    def _save_as_rejected(self, cut: NDArray[np.uint8]) -> None:
        rejected = (
            self.dst.with_name(f"{self.dst.name}(rejected)") / self.ep_dir.stem
        )
        rejected.mkdir(parents=True, exist_ok=True)
        f = len(list(rejected.glob("*.jpg")))
        Image.fromarray(cut).save(rejected / f"{f}.jpg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--src", default="./girls-trial", type=click.STRING)
@click.option("--dst", default="./result", type=click.STRING)
@click.option(
    "--hlim",
    default=(100, 5_000),
    type=click.Tuple([click.INT, click.INT]),
    help="(min_height, max_height)",
)
@click.option("--tol", default=50.0, type=click.FLOAT)
def main(src: str, dst: str, hlim: tuple[int, int], tol: float) -> None:
    min_height, max_height = hlim
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.mkdir(exist_ok=True)

    for ep_dir in tqdm(list(src_p.glob("*"))):
        if not ep_dir.is_dir():
            continue
        (dst_p / ep_dir.stem).mkdir(exist_ok=True)
        extractor = CutExtractor(
            ChunkImageLoader(ep_dir),
            min_height,
            max_height,
            color_tol=tol,
            dst=dst_p,
        )
        for i, cut in enumerate(extractor.extract()):
            assert cut.height >= min_height
            assert cut.height <= max_height
            cut.save(dst_p / ep_dir.stem / f"{i:03}.jpg")


if __name__ == "__main__":
    main()
