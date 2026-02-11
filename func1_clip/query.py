from clip_retrieval.clip_client import ClipClient

PORT=13131
INDICE_PATH="index_h14"

client = ClipClient(url=f"http://localhost:{PORT}/knn-service", indice_name=INDICE_PATH)

results = client.query(text="bloody")
results = client.query(text="bloody", image="../crawling/738487/0001/001.jpg")

results
