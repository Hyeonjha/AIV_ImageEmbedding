import numpy as np
import pickle
import time
from statistics import mean, stdev
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from ImageEmbedding import ImageEmbedding, HNSWIndex  # 모델 정의를 포함한 모듈에서 가져옴

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def load_hnsw_index(session):
    hnsw_index_record = session.query(HNSWIndex).first()
    hnsw_index = pickle.loads(hnsw_index_record.index_data)
    return hnsw_index

def find_similar_images(input_embedding, session, hnsw_index, top_n=5):
    labels, distances = hnsw_index.knn_query(input_embedding, k=top_n)  # knn_query : 근사 최근접 이웃 검색
    nearest_ids = labels[0].tolist()

    query = session.query(ImageEmbedding).filter(ImageEmbedding.id.in_(nearest_ids))
    results = query.all()
    similarities = [(res.image_path, res.label, 1 - distances[0][i]) for i, res in enumerate(results)]
    return similarities

def search_fake_vectors(session, hnsw_index, num_vectors=100000, dim=1000, top_n=5):  # num_vectors=100000
    times = []
    for i in range(num_vectors):
        fake_vector = np.random.rand(dim).tolist()
        start_time = time.time()
        _ = find_similar_images(fake_vector, session, hnsw_index, top_n=top_n)
        end_time = time.time()
        times.append(end_time - start_time)
        if (i + 1) % 10000 == 0:
            print(f"Search - {i + 1} vectors searched")
            avg_time = mean(times[-10000:])
            stddev_time = stdev(times[-10000:])
            print(f"Average Time: {avg_time:.6f} seconds, Standard Deviation: {stddev_time:.6f} seconds")
    return times

if __name__ == "__main__":
    hnsw_index = load_hnsw_index(session)

    times = search_fake_vectors(session, hnsw_index)
    avg_time = mean(times)
    stddev_time = stdev(times)
    print(f"Search - Total Average Time: {avg_time:.6f} seconds, Standard Deviation: {stddev_time:.6f} seconds")

    # 검색 성능 평가를 위해 몇 개의 검색 결과 출력
    sample_vector = np.random.rand(1000).tolist()
    sample_results = find_similar_images(sample_vector, session, hnsw_index)
    print("Sample search results:")
    for image_path, label, cosine_similarity in sample_results:
        print(f"Image Path: {image_path}, Label: {label}, Similarity: {cosine_similarity:.4f}")
