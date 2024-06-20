import time
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from ImageEmbedding import ImageEmbedding, Base
from datasketch import MinHash, MinHashLSH

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def generate_random_embeddings(num_embeddings, dim=1000):
    return [np.random.rand(dim).tolist() for _ in range(num_embeddings)]

def lsh_setup(embeddings, num_perm=128):
    lsh = MinHashLSH(threshold=0.8, num_perm=num_perm)
    for i, embedding in enumerate(embeddings):
        m = MinHash(num_perm=num_perm)
        for d in embedding:
            m.update(str(d).encode('utf8'))
        lsh.insert(f"embedding_{i}", m)
    return lsh

def find_similar_lsh(lsh, query_embedding, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for d in query_embedding:
        m.update(str(d).encode('utf8'))
        
    return lsh.query(m)

def measure_search_time(session, lsh, threshold=0.8, top_n=5, total_tests=100000, dim=1000, num_perm=128):
    search_times = []
    random_embeddings = [np.random.rand(dim).tolist() for _ in range(total_tests)]

    for idx, random_embedding in enumerate(random_embeddings):
        start_time = time.time()
        find_similar_lsh(lsh, random_embedding, num_perm=num_perm)
        end_time = time.time()
        search_times.append(end_time - start_time)
        
        if (idx + 1) % 10000 == 0:
            search_times_np = np.array(search_times)
            print(f"Search times for {idx + 1} queries: {search_times_np.mean()} ± {search_times_np.std()} seconds")
            search_times = []  # Reset for the next batch

    return search_times_np.mean(), search_times_np.std()

if __name__ == "__main__":
    # 임베딩을 생성하고 LSH에 삽입하는 단계는 생략, 이 단계는 postInsert.py에서 수행됨
    total_embeddings = 100000
    num_perm = 128

    embeddings = generate_random_embeddings(total_embeddings)
    lsh = lsh_setup(embeddings, num_perm=num_perm)

    measure_search_time(session, lsh, total_tests=100000, num_perm=num_perm)
