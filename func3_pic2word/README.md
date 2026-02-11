먼저 `pic2word_model.pt` 파일을 [여기](https://drive.google.com/file/d/1IxRi2Cj81RxMu0ViT4q4nkfyjbSHm1dF/view)서 다운 받아주세요.

```bash
docker build --tag pic2word .
```

```bash
docker run --rm -ti \
    --name pic2word \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ./code:/code \
    -v ./pic2word_model.pt:/pic2word_model.pt \
    -v ./cache:/root/.cache \
    -w /code \
    -p 8000:8000 \
    pic2word fastapi run main.py --host 0.0.0.0 --port 8000
```



