import requests

res = requests.post(
    "http://localhost:8000/rerank",
    json={
        "query": "there exists 박가을",
        "target_images": [
            {"ep": "0001", "cut": "001.jpg"},
            {"ep": "0001", "cut": "003.jpg"},
            {"ep": "0001", "cut": "005.jpg"}

        ]
    })
print(res.json())


res = requests.post(
    "http://localhost:8000/retrieve",
    json={"query": "there exists 박가을"}
)
print(res.json())
