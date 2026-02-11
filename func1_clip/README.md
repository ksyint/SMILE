```sh
git clone https://github.com/rom1504/clip-retrieval.git
```

```sh
conda create -n clipretrival python=3.10
conda activate clipretrival

conda install ipykernel pillow
pip install -r clip-retrieval/requirements-test.txt
pip install clip-retrieval img2dataset
```


```sh
clip-retrieval inference --input_dataset ../dataset_shared2/orig-result --output_folder embeddings_shared2_default --enable_text False

# --clip_model (default: ViT-B/32)
## open_clip:ViT-B-32/laion2b_s34b_b79k : 기본 모델보다 비슷한 정도
## open_clip:ViT-B-32-quickgelu : 성능이 별로...
## hf_clip:Bingsu/clip-vit-large-patch14-ko : 성능이 별로...

## --clip_model hf_clip:apple/DFN5B-CLIP-ViT-H-14-378
## hf_clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K : 생각보다 좋은 것 같다.

# --use_mclip True
# --mclip_model sentence-transformers/clip-ViT-B-32-multilingual-v1
## 
```

```sh
clip-retrieval index --embeddings_folder ./embeddings_shared2_default --index_folder index_shared2_default
```

```sh
echo '{"index_default": "index_shared2_default"}' > indices_paths.json
clip-retrieval back --port 13131 --indices-paths indices_paths.json --provide_safety_model False --provide_violence_detector False 
```

```sh
echo '{"index_h14": "index_shared2_h14"}' > indices_paths.json
clip-retrieval back --port 13131 --indices-paths indices_paths.json --clip_model hf_clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K --provide_safety_model False --provide_violence_detector False --enable_mclip_option False --provide_aesthetic_embeddings False

# --clip_model
## open_clip:ViT-B-32/laion2b_s34b_b79k
## open_clip:ViT-B-32-quickgelu
## hf_clip:Bingsu/clip-vit-large-patch14-ko

# --enable_mclip_option False
# --provide_aesthetic_embeddings False
```