"""
Composed Image Retrieval (Pic2Word) 모듈
- 캐릭터 이미지 + 텍스트 프롬프트를 결합한 이미지 검색
- CLIP 모델 + IM2TEXT 변환기를 사용한 파이프라인
- FastAPI 서버 모드 및 라이브러리 모드 지원

Merges: composed-image-retrieval-main/code/main.py,
        gradio_demo_api-main/func3.py, func3_pic2word/
"""

import os
import glob
import ast
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from pydantic import BaseModel
from googletrans import Translator


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Composed Image Search Pipeline
# ---------------------------------------------------------------------------

class ComposedImageSearchPipeline:
    """
    텍스트 프롬프트 내 '*' 토큰을 캐릭터 이미지 임베딩으로 대체하여
    composed query를 만든 뒤 타겟 이미지들과 유사도를 계산한다.
    """

    def __init__(
        self,
        model,        # CLIP model
        img2text,     # IM2TEXT model
        target_image_paths: list[Path],
        transform: Callable[[Image.Image], torch.Tensor],
        id_split_token: Optional[int] = None,
    ) -> None:
        self.model = model
        self.img2text = img2text
        self.target_image_paths = target_image_paths
        self.transform = transform
        self._id_split = id_split_token

    @property
    def id_split(self) -> int:
        if self._id_split is None:
            from third_party.open_clip.clip import tokenize
            self._id_split = tokenize(["*"])[0][1]
        return self._id_split

    def load_target_images(self) -> torch.Tensor:
        tensors = [self.transform(Image.open(p)) for p in self.target_image_paths]
        return torch.stack(tensors, dim=0).cuda(non_blocking=True)

    def get_tgt_img_feats(self) -> torch.Tensor:
        target_images = self.load_target_images()
        feats = self.model.encode_image(target_images)
        return _normalize(feats)

    def get_qry_img_feat(self, qry_img: Image.Image) -> torch.Tensor:
        from model.clip import _transform
        transform = _transform(self.model.visual.input_resolution)
        qry = transform(qry_img).unsqueeze(0).cuda(non_blocking=True)
        return self.img2text(self.model.encode_image(qry))

    def tokenize(self, prompt: str) -> torch.Tensor:
        from third_party.open_clip.clip import tokenize
        tokens = tokenize(prompt)
        assert self.id_split in tokens
        return tokens.cuda(non_blocking=True)

    @torch.no_grad()
    def search(
        self, qry_img: Image.Image, prompt: str
    ) -> torch.Tensor:
        tgt_feats = self.get_tgt_img_feats()
        qry_feat = self.get_qry_img_feat(qry_img)
        tokens = self.tokenize(prompt)
        composed = self.model.encode_text_img_vis(
            tokens, qry_feat, split_ind=self.id_split,
        )
        composed = _normalize(composed)
        sim = composed @ tgt_feats.T
        return sim.squeeze(0)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_models(
    model_id: str = "ViT-L/14",
    ckpt: str | Path = "./pic2word_model.pt",
):
    """
    CLIP + IM2TEXT 모델을 로드한다.

    Returns:
        (model, img2text, transform)
    """
    from model.clip import load
    from model.model import IM2TEXT

    model, _, preprocess_val = load(model_id, jit=False)
    img2text = IM2TEXT(
        embed_dim=model.embed_dim,
        output_dim=model.token_embedding.weight.shape[1],
    )
    model.cuda()
    img2text.cuda()
    img2text.half()

    checkpoint = torch.load(ckpt, map_location="cuda:0")
    sd = {k.removeprefix("module."): v
          for k, v in checkpoint["state_dict"].items()}
    sd_img2text = {k.removeprefix("module."): v
                   for k, v in checkpoint["state_dict_img2text"].items()}
    model.load_state_dict(sd)
    img2text.load_state_dict(sd_img2text)

    return model, img2text, preprocess_val


# ---------------------------------------------------------------------------
# Character Helpers
# ---------------------------------------------------------------------------

def load_character_embeddings(
    char_dir: str = "./character_image_embeddings",
) -> dict[str, Image.Image]:
    return {
        Path(p).stem: Image.open(p)
        for p in sorted(glob.glob(os.path.join(char_dir, "*.png")))
    }


def get_composed_query(
    query: str,
    character_dict: dict[str, Image.Image],
) -> tuple[str, Image.Image]:
    """쿼리에서 캐릭터 이름을 '*'로 대체하고 해당 캐릭터 이미지를 반환한다."""
    for name, img in character_dict.items():
        if name in query:
            return query.replace(name, "*"), img
    raise ValueError(f"쿼리에서 캐릭터 이름을 찾을 수 없습니다: {query}")


# ---------------------------------------------------------------------------
# High-level Search
# ---------------------------------------------------------------------------

def do_retrieve(
    query: str,
    target_paths: list[str] | str,
    k: int = 5,
    model=None,
    img2text=None,
    transform=None,
    character_dict: Optional[dict[str, Image.Image]] = None,
    base_dir: str = "./result",
) -> tuple[str, Image.Image, list[dict]]:
    """
    Composed image retrieval을 수행한다.

    Args:
        query: 검색 쿼리 (캐릭터 이름 포함)
        target_paths: 검색 대상 이미지 경로 리스트
        k: 반환할 상위 결과 수
    """
    if isinstance(target_paths, str):
        target_paths = ast.literal_eval(target_paths)

    if character_dict is None:
        character_dict = load_character_embeddings()

    paths = [os.path.join(base_dir, p) for p in target_paths]
    pipe = ComposedImageSearchPipeline(
        model, img2text,
        target_image_paths=[Path(p) for p in paths],
        transform=transform,
    )

    composed_query, character_img = get_composed_query(query, character_dict)

    # 한국어 → 영어 번역
    translator = Translator()
    composed_query_en = translator.translate(
        composed_query, src="ko", dest="en"
    ).text

    similarities = pipe.search(character_img, prompt=composed_query_en)
    similarities = similarities.cpu().tolist()
    results = sorted(
        zip(similarities, pipe.target_image_paths),
        key=lambda x: x[0],
        reverse=True,
    )[:k]

    outputs = []
    for sim, img_path in results:
        img_path_str = str(img_path)
        episode, basename = img_path_str.split("/")[-2:]
        num, _ = os.path.splitext(basename)
        outputs.append({
            "image_path": f"{episode}/{num}.jpg",
            "episode": int(episode),
            "num": int(num),
            "similarity": sim,
        })

    return composed_query_en, character_img, outputs


# ---------------------------------------------------------------------------
# FastAPI Server
# ---------------------------------------------------------------------------

class ImageInfo(BaseModel):
    ep: str
    cut: str


class RerankRequest(BaseModel):
    query: str
    target_images: list[ImageInfo]


class RetrieveRequest(BaseModel):
    query: str


def create_fastapi_app(
    model_id: str = "ViT-L/14",
    ckpt: str = "./pic2word_model.pt",
    char_dir: str = "./character_image_embeddings",
    result_dir: str = "./result",
):
    """FastAPI 앱을 생성한다."""
    from fastapi import FastAPI

    model, img2text, transform = load_models(model_id, ckpt)
    character_dict = load_character_embeddings(char_dir)
    all_target_paths = sorted(Path(result_dir).glob("*/*.jpg"))

    app = FastAPI(title="Composed Image Retrieval API")

    retrieve_pipe = ComposedImageSearchPipeline(
        model, img2text, target_image_paths=all_target_paths, transform=transform,
    )

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        query, char_img = get_composed_query(req.query, character_dict)
        sims = retrieve_pipe.search(char_img, prompt=query)
        sims = sims.cpu().tolist()
        ranked = sorted(
            zip(sims, retrieve_pipe.target_image_paths),
            key=lambda x: x[0], reverse=True,
        )[:10]
        return [
            {"score": s, "ep": p.parent.stem, "cut": p.name}
            for s, p in ranked
        ]

    @app.post("/rerank")
    def rerank(req: RerankRequest):
        paths = [Path(result_dir) / t.ep / t.cut for t in req.target_images]
        pipe = ComposedImageSearchPipeline(
            model, img2text, target_image_paths=paths, transform=transform,
        )
        query, char_img = get_composed_query(req.query, character_dict)
        sims = pipe.search(char_img, prompt=query)
        sims = sims.cpu().tolist()
        ranked = sorted(
            zip(sims, paths), key=lambda x: x[0], reverse=True,
        )[:10]
        return [
            {"score": s, "ep": p.parent.stem, "cut": p.name}
            for s, p in ranked
        ]

    return app
