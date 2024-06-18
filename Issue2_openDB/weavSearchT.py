import numpy as np
import time
import weaviate

# Weaviate 클라이언트 설정
client = weaviate.Client(url="http://localhost:8080")

# Weaviate certainty 값을 코사인 유사도로 변환하는 함수
def weaviate_certainty_to_cosine(certainty):
    return 2 * certainty - 1

# 벡터 생성 함수 (검색을 위한)
def generate_random_vector(dim=1000):
    return np.random.rand(dim).astype(np.float32)

# 검색 성능 테스트
def search_fake_vectors(num_searches=100000, batch_size=10000, dim=1000):
    search_times = []
    for i in range(num_searches):
        fake_query_vector = generate_random_vector(dim=dim)
        start_time = time.time()
        query_result = client.query.get("FakeImageEmbedding", ["image_path", "label", "_additional {certainty}"])\
            .with_near_vector({"vector": fake_query_vector})\
            .with_limit(5)\
            .do()
        end_time = time.time()
        search_times.append(end_time - start_time)

        # 10,000개 단위로 결과 출력
        if (i + 1) % batch_size == 0:
            batch_mean = np.mean(search_times[-batch_size:])
            batch_std = np.std(search_times[-batch_size:])
            print(f"Searched {i + 1} vectors - Batch mean time: {batch_mean:.4f}, Batch std time: {batch_std:.4f}")

    total_mean = np.mean(search_times)
    total_std = np.std(search_times)

    print(f"Total Search - Mean time: {total_mean:.4f}, Std time: {total_std:.4f}")

if __name__ == "__main__":
    search_fake_vectors()
