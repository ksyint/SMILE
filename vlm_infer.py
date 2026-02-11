"""
VLM (Vision-Language Model) 추론 모듈
- KoLLaVA 기반 이미지 설명 생성
- 웹툰 장면에 대한 한국어 설명/캡션 생성

Merges: kollava-main/code/infer.py
"""

from io import BytesIO
from typing import Optional

import click
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor


# ---------------------------------------------------------------------------
# Image Loading
# ---------------------------------------------------------------------------

def load_image(image_file: str) -> Image.Image:
    """로컬 경로 또는 URL로부터 이미지를 로드한다."""
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


# ---------------------------------------------------------------------------
# KoLLaVA Inference Pipeline
# ---------------------------------------------------------------------------

class KoLLaVAInferencePipeline:
    """
    KoLLaVA 모델을 사용하여 이미지에 대한 한국어 설명을 생성한다.

    사용 예:
        pipe = KoLLaVAInferencePipeline()
        result = pipe.inference("이 장면에 대해 설명해주세요.", "image.png")
    """

    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"

    def __init__(
        self,
        model_name: str = "tabtoyou/KoLLaVA-KoVicuna-7b",
    ):
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.model import LlavaLlamaForCausalLM
        from llava.model.utils import KeywordsStoppingCriteria

        self.model_name = model_name
        self._conv_templates = conv_templates
        self._SeparatorStyle = SeparatorStyle
        self._KeywordsStoppingCriteria = KeywordsStoppingCriteria

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
        )

        # 모델
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_cache=True,
        ).cuda()

        # 이미지 프로세서
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.model.config.mm_vision_tower,
            torch_dtype=torch.float16,
        )

        # 특수 토큰 추가
        self.tokenizer.add_tokens(
            [self.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
        )
        self.tokenizer.add_tokens(
            [self.DEFAULT_IM_START_TOKEN, self.DEFAULT_IM_END_TOKEN],
            special_tokens=True,
        )

        # 비전 타워 초기화
        vision_tower = self.model.get_model().vision_tower[0]
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        self.model.get_model().vision_tower[0] = vision_tower

        # 이미지 토큰 관련 설정
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids(
            [self.DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
        vision_config.use_im_start_end = getattr(
            self.model.config, "mm_use_im_start_end", False
        )
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = self.tokenizer.convert_tokens_to_ids(
            [self.DEFAULT_IM_START_TOKEN, self.DEFAULT_IM_END_TOKEN]
        )
        self.image_token_len = (
            vision_config.image_size // vision_config.patch_size
        ) ** 2

    def inference(
        self,
        query: str,
        image_file: str,
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        이미지에 대해 질의하고 응답을 생성한다.

        Args:
            query: 질의 텍스트 (한국어)
            image_file: 이미지 경로 또는 URL
            temperature: 생성 온도 (낮을수록 결정적)
            max_new_tokens: 최대 생성 토큰 수

        Returns:
            모델 응답 텍스트
        """
        image = load_image(image_file)

        # 프롬프트 구성
        qs = (
            query
            + "\n"
            + self.DEFAULT_IM_START_TOKEN
            + self.DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
            + self.DEFAULT_IM_END_TOKEN
        )

        conv = self._conv_templates["multimodal"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])

        image_tensor = self.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = (
            conv.sep
            if conv.sep_style != self._SeparatorStyle.TWO
            else conv.sep2
        )
        stopping_criteria = self._KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff > 0:
            print(f"[Warning] {n_diff} output_ids differ from input_ids")

        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()

        return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--src", "-s", default="./images/cut5.png",
              help="이미지 경로 또는 URL")
@click.option("--query", "-q", default="이 장면에 대해 설명해주세요.",
              help="질의 텍스트")
@click.option("--model_name", "-m", default="tabtoyou/KoLLaVA-KoVicuna-7b")
def main(src: str, query: str, model_name: str):
    pipe = KoLLaVAInferencePipeline(model_name=model_name)
    result = pipe.inference(query, src)
    print(result)


if __name__ == "__main__":
    main()
