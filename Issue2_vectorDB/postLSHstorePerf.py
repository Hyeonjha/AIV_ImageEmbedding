import time
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ImageEmbedding import ImageEmbedding, Base
from datasketch import MinHash, MinHashLSH

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def initialize_database(session):
    session.query(ImageEmbedding).delete()
    session.commit()
    print("Database initialized: All records have been deleted.")

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

def save_random_embeddings(num_embeddings, session, lsh, offset=0, num_perm=128):
    embeddings = generate_random_embeddings(num_embeddings)
    insert_times = []

    for i, embedding in enumerate(embeddings):
        image_embedding = ImageEmbedding(
            image_path=f"fake_path_{i + offset}.jpg",  # 고유한 image_path 생성
            label=f"fake_label_{i + offset}",
            embedding=embedding,
            embedding_vector=embedding  # pgvector 저장
        )
        start_time = time.time()
        session.add(image_embedding)
        session.commit()
        end_time = time.time()
        insert_times.append(end_time - start_time)

        m = MinHash(num_perm=num_perm)
        for d in embedding:
            m.update(str(d).encode('utf8'))
        lsh.insert(f"embedding_{i + offset}", m)

    insert_times_np = np.array(insert_times)
    print(f"Insert times over {num_embeddings} embeddings: {insert_times_np.mean()} ± {insert_times_np.std()} seconds")
    return insert_times_np.mean(), insert_times_np.std()

if __name__ == "__main__":
    initialize_database(session)  # 데이터베이스 초기화

    total_embeddings = 100000
    batch_size = 10000
    num_perm = 128

    lsh = MinHashLSH(threshold=0.8, num_perm=num_perm)

    for batch in range(0, total_embeddings, batch_size):
        print(f"Inserting {batch + batch_size} embeddings...")
        save_random_embeddings(batch_size, session, lsh, offset=batch, num_perm=num_perm)
