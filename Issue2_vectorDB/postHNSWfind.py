import hnswlib
import numpy as np
from PIL import Image
from Img2Vec import Img2Vec
import psycopg2

# PostgreSQL 설정
conn = psycopg2.connect("dbname=postgres user=postgres password=aiv11011 host=localhost")
cur = conn.cursor()

def find_similar_images(input_image_path, top_n=5):
    img = Image.open(input_image_path).convert('RGB')
    img2vec = Img2Vec()
    input_embedding = img2vec.get_vec(img)

    dim = 1000  # 벡터 차원 수
    index = hnswlib.Index(space='cosine', dim=dim)
    index.load_index("hnsw_index.bin")

    labels, distances = index.knn_query(input_embedding, k=top_n)
    nearest_ids = labels[0].tolist()  # numpy.ndarray를 Python 리스트로 변환

    cur.execute("SELECT image_path, label FROM image_embeddings WHERE id = ANY(%s)", (nearest_ids,))
    results = cur.fetchall()

    similarities = [(res[0], res[1], 1 - distances[0][i]) for i, res in enumerate(results)]

    return similarities

if __name__ == "__main__":
    input_image_path = 'data-gatter/test/damage_378110.jpg'
    similar_images = find_similar_images(input_image_path)
    for image_path, label, similarity in similar_images:
        print(f"Image Path: {image_path}, Label: {label}, Similarity: {similarity:.4f}")
