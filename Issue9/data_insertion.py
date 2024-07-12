import os
import weaviate
import uuid
import numpy as np
import time

# 환경 변수 설정
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# Weaviate 클라이언트 설정
client = weaviate.Client(WEAVIATE_URL)

# Weaviate 스키마 설정
schema = {
    "classes": [
        {
            "class": "ImageEmbedding",
            "description": "A class to store image embeddings",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "uuid",
                    "dataType": ["string"],
                    "description": "The UUID of the image"
                },
                {
                    "name": "embedding",
                    "dataType": ["number[]"],
                    "description": "The embedding vector of the image"
                }
            ]
        }
    ]
}
if not client.schema.contains(schema):
    client.schema.create(schema)

def generate_random_vector(dim=1000):
    return np.random.rand(dim).astype(np.float32).tolist()

def insert_fake_vectors(num_vectors=100000, batch_size=1000, method='random', dim=1000):
    for i in range(0, num_vectors, batch_size):
        batch = []
        for j in range(batch_size):
            if method == 'random':
                fake_vector = generate_random_vector(dim=dim)
            else:
                raise ValueError("Invalid method. Use 'random'.")
            
            image_id = str(uuid.uuid4())
            
            data_object = {
                "uuid": image_id,
                "embedding": fake_vector
            }
            batch.append((data_object, fake_vector))

        # Weaviate에 배치로 데이터 삽입
        with client.batch as batch_client:
            for data_object, vector in batch:
                try:
                    batch_client.add_data_object(data_object, "ImageEmbedding", vector=vector)
                except weaviate.exceptions.WeaviateBaseException as e:
                    print(f"Error inserting data object: {e}")
                    continue
        
        print(f"Inserted {i + batch_size} vectors")

if __name__ == "__main__":
    insert_fake_vectors()
