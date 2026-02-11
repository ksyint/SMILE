- http://1.215.235.253:17000/demo
- http://1.215.235.253:17000/docs

### Installation
```sh
conda create -n demo python=3.10
conda activate demo

conda install ipykernel pillow
pip install clip-retrieval img2dataset

pip install googletrans-py
pip install langchain langchain-huggingface langchain-community
pip install openai==0.28

```
참고
- https://github.com/ShivangKakkar/googletrans

### Run clip-retrieval
```sh
tmux new -s vvave_clip_retrieval

conda activate clipretrival
clip-retrieval back --port 13131 --indices-paths indices_paths.json --clip_model hf_clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K --provide_safety_model False --provide_violence_detector False --enable_mclip_option False --provide_aesthetic_embeddings False
```

### Run demo
```sh
tmux new -s vvave_demo_api

conda activate clipretrival
python demo.py --host 0.0.0.0 --port 7000 --workers 1

# 로컬 테스트
python demo.py --port 7777 --workers 1
```