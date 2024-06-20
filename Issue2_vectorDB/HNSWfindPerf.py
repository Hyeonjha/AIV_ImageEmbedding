import time
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import hnswlib
from ImageEmbedding import ImageEmbedding, Base

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def generate_random_embeddings(num_embeddings, dim=1000):
    return [np.random.rand(dim).tolist() for _ in range(num_embeddings)]

def find_similar_embeddings(random_embedding, index, top_n=5):
    labels, distances = index.knn_query(random_embedding, k=top_n)
    labels = labels[0].tolist()  # numpy.uint64를 Python 정수로 변환
    distances = distances[0].tolist()
    return labels, distances

def measure_search_time(session, total_tests=100000, dim=1000):
    batch_size = 10000
    index = hnswlib.Index(space='cosine', dim=dim)
    index.load_index("hnsw_index.bin")

    for batch_start in range(0, total_tests, batch_size):
        search_times = []
        random_embeddings = generate_random_embeddings(batch_size, dim)

        for idx, random_embedding in enumerate(random_embeddings):
            start_time = time.time()
            labels, distances = find_similar_embeddings(random_embedding, index, top_n=5)
            end_time = time.time()
            search_times.append(end_time - start_time)

            # 검색 결과의 예시 출력 (첫 번째 배치의 첫 5개의 검색 결과만 출력)
            if batch_start == 0 and idx < 5:
                print(f"Search results for test {idx + 1}:")
                # cosine similarity = 1 - cosine distance (distance 작을수록 유사도 높음)
                for label, distance in zip(labels, distances):
                    result = session.query(ImageEmbedding).filter_by(id=int(label)).first()
                    if result:
                        print(f"  Image Path: {result.image_path}, Label: {result.label}, Distance: {1 - distance}")
                    else:
                        print("  No result found for id:", int(label))

        search_times_np = np.array(search_times)
        mean_time = search_times_np.mean()
        std_time = search_times_np.std()
        print(f"Search times for batch {batch_start + batch_size}: {mean_time} ± {std_time} seconds")

if __name__ == "__main__":
    # 검색 시간 측정
    measure_search_time(session, total_tests=100000)
