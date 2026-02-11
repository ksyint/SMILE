from pathlib import Path
import torch
from PIL import Image
from pathlib import Path
from typing import Callable
import ast
from googletrans import Translator
import os, glob
import sys
sys.path.append('./func3_pic2word/code')
from model.clip import _transform, load
from model.model import CLIP, IM2TEXT
from third_party.open_clip.clip import tokenize, _transform

def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)

SAMPLE_DIR = "./func3_pic2word/code/result"
ALL_DIR = "../dataset_shared2/orig-result"

class ComposedImageSearchPipeline:
    ID_SPLIT = tokenize(["*"])[0][1]

    def __init__(self,
                 model: CLIP,
                 img2text: IM2TEXT,
                 target_image_paths: list[Path],
                 transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.model = model
        self.img2text = img2text
        self.target_image_paths = target_image_paths
        self.transform = transform

    def load_target_images(self) -> torch.Tensor:
        img_tensors = [self.transform(Image.open(p))
                       for p in self.target_image_paths]
        img_tensors = torch.stack(img_tensors, dim=0).cuda(non_blocking=True)
        return img_tensors

    def get_tgt_img_feats(self) -> torch.Tensor:
        target_images = self.load_target_images()
        tgt_image_features = self.model.encode_image(target_images)
        tgt_image_features = _normalize(tgt_image_features)
        return tgt_image_features

    def get_qry_img_feat(self, qry_img: Image.Image) -> torch.Tensor:
        transform = _transform(self.model.visual.input_resolution)
        qry_img = transform(qry_img)  # type: ignore
        qry_img = qry_img.unsqueeze(0).cuda(non_blocking=True)  # type: ignore
        qry_img_feature = self.img2text(self.model.encode_image(qry_img))
        return qry_img_feature

    def tokenize(self, prompt: str) -> torch.Tensor:
        text_tokens = tokenize(prompt)
        assert self.ID_SPLIT in text_tokens
        text_tokens = text_tokens.cuda(non_blocking=True)

        return text_tokens

    @torch.no_grad()
    def search(self, qry_img: Image.Image, prompt: str) -> torch.Tensor:
        tgt_image_features = self.get_tgt_img_feats()
        qry_img_feature = self.get_qry_img_feat(qry_img)
        text_tokens = self.tokenize(prompt)
        composed_qry_feature = self.model.encode_text_img_vis(
            text_tokens,
            qry_img_feature,
            split_ind=self.ID_SPLIT,  # type: ignore
        )
        composed_qry_feature = _normalize(composed_qry_feature)

        similarity = composed_qry_feature @ tgt_image_features.T
        similarity = similarity.squeeze(0)
        return similarity

def load_models(model_id: str = 'ViT-L/14',
                ckpt: str | Path = './func3_pic2word/pic2word_model.pt',
                ) -> tuple[CLIP, IM2TEXT, Callable[[Image.Image], torch.Tensor]]:
    model, preprocess_train, preprocess_val = load(model_id, jit=False)
    img2text = IM2TEXT(embed_dim=model.embed_dim,
                       output_dim=model.token_embedding.weight.shape[1])
    model.cuda()
    img2text.cuda()
    img2text.half()
    checkpoint = torch.load(ckpt, map_location="cuda:0")
    sd = checkpoint["state_dict"]
    sd_img2text = checkpoint["state_dict_img2text"]
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
    model.load_state_dict(sd)
    img2text.load_state_dict(sd_img2text)

    transform = preprocess_val
    return model, img2text, transform


model, img2text, transform = load_models()


CHARACTER_EMBEDDING_DICT: dict[str, Image.Image] = {
    p.stem: Image.open(p)
    for p in Path("./func3_pic2word/code/character_image_embeddings").glob("*.png")
}

MAIN_CHARACTERS: list[str] = [*CHARACTER_EMBEDDING_DICT.keys()]

def get_composed_query(query: str) -> tuple[str, Image.Image]:
    for character_name, character_img in CHARACTER_EMBEDDING_DICT.items():
        if character_name in query:
            query = query.replace(character_name, "*")
            return query, character_img
    raise TypeError()

# type : sample or all
def get_target_paths(type: str) :
    if type == 'sample' :
        target_paths = sorted(glob.glob(f'{SAMPLE_DIR}/**/*.jpg', recursive=True))
        target_paths = ['/'.join(sample_path.split('/')[-2:]) for sample_path in target_paths]
    else :
        target_paths = sorted(glob.glob(f'{ALL_DIR}/**/*.jpg', recursive=True))
        target_paths = ['/'.join(all_path.split('/')[-2:]) for all_path in target_paths]
    return target_paths

def do_retrieve(query: str, target_paths: str, k: int) :
    target_paths = ast.literal_eval(target_paths)

    paths = [os.path.join(ALL_DIR, target_path) for target_path in target_paths]
    retrieve_pipe = ComposedImageSearchPipeline(model, img2text,
                                                target_image_paths=paths,
                                                transform=transform)

    query, character_img = get_composed_query(query)

    # 한국어 -> 영어 번역
    translator = Translator()
    query = translator.translate(query, src='ko', dest='en').text

    similarities = retrieve_pipe.search(character_img, prompt=query)
    similarities = similarities.cpu().tolist()
    results = sorted(zip(similarities, retrieve_pipe.target_image_paths),
                     key=lambda x: x[0],
                     reverse=True)[:k]

    outputs = []
    for sim, img_path in results :
        info = {}

        episode, basename = img_path.split('/')[-2:]
        num, _ = os.path.splitext(basename)

        info['image_path'] = f"{episode}/{num}.jpg"
        info['episode'] = int(episode)
        info['num'] = int(num)
        info['similarity'] = sim

        outputs.append(info)

    return query, character_img, outputs