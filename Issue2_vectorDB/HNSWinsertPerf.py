import time
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ImageEmbedding import ImageEmbedding, Base
import hnswlib

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

def save_random_embeddings(num_embeddings, session, index, offset=0):
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
        
        # 데이터베이스에 저장된 실제 id를 사용하여 HNSW 인덱스에 추가
        index.add_items([embedding], [image_embedding.id])
        insert_times.append(end_time - start_time)

    insert_times_np = np.array(insert_times)
    mean_time = insert_times_np.mean()
    std_time = insert_times_np.std()
    print(f"Insert times over {num_embeddings} embeddings: {mean_time} ± {std_time} seconds")
    return mean_time, std_time

if __name__ == "__main__":
    initialize_database(session)  # 데이터베이스 초기화

    total_embeddings = 100000
    batch_size = 10000
    dim = 1000
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=total_embeddings, ef_construction=200, M=16)

    for batch in range(0, total_embeddings, batch_size):
        print(f"Inserting {batch + batch_size} embeddings...")
        save_random_embeddings(batch_size, session, index, offset=batch)

    index.save_index("hnsw_index.bin")
    print("HNSW index created and saved to 'hnsw_index.bin'")
