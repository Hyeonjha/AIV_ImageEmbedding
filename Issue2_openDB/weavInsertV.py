import numpy as np
import time
import weaviate

# Weaviate 클라이언트 설정
client = weaviate.Client(url="http://localhost:8080")

# 스키마 정의
class_obj = {
    "class": "FakeImageEmbedding",
    "vectorizer": "none",
    "properties": [
        {
            "name": "image_path",
            "dataType": ["string"]
        },
        {
            "name": "label",
            "dataType": ["string"]
        }
    ]
}

# 클래스 존재 여부 확인 및 생성
def ensure_class_exists(client, class_obj):
    try:
        existing_classes = client.schema.get()['classes']
        existing_class_names = [cls['class'] for cls in existing_classes]
        if class_obj['class'] not in existing_class_names:
            client.schema.create_class(class_obj)
        else:
            print(f"Class {class_obj['class']} already exists.")
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        print(f"Error checking/creating class: {e}")

ensure_class_exists(client, class_obj)

# 벡터 생성 함수
def generate_random_vector(dim=1000):
    return np.random.rand(dim).astype(np.float32)

# 임의의 벡터 생성 및 데이터 삽입
def insert_fake_vectors(num_vectors=100000, batch_size=10000, method='random', dim=1000):
    insert_times = []
    for i in range(num_vectors):
        if method == 'random':
            fake_vector = generate_random_vector(dim=dim)
        else:
            raise ValueError("Invalid method. Use 'random'.")

        data_object = {
            "image_path": f"fake_image_{i}.jpg",
            "label": "fake"
        }
        start_time = time.time()
        client.data_object.create(data_object, "FakeImageEmbedding", vector=fake_vector)
        end_time = time.time()
        insert_times.append(end_time - start_time)

        # 10,000개 단위로 결과 출력
        if (i + 1) % batch_size == 0:
            batch_mean = np.mean(insert_times[-batch_size:])
            batch_std = np.std(insert_times[-batch_size:])
            print(f"Inserted {i + 1} vectors - Batch mean time: {batch_mean:.4f}, Batch std time: {batch_std:.4f}")

    total_mean = np.mean(insert_times)
    total_std = np.std(insert_times)

    print(f"Total Insert - Mean time: {total_mean:.4f}, Std time: {total_std:.4f}")

if __name__ == "__main__":
    insert_fake_vectors()
