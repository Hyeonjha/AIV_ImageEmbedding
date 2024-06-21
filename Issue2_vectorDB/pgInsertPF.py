import os
import hnswlib
import numpy as np
import pickle
import time
from statistics import mean, stdev
from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker 

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

Base = declarative_base()

class ImageEmbedding(Base):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True)
    image_path = Column(String, nullable=False)
    label = Column(String, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    embedding_vector = Column(ARRAY(Float), nullable=False)

class HNSWIndex(Base):
    __tablename__ = 'hnsw_index'
    id = Column(Integer, primary_key=True)
    index_data = Column(LargeBinary, nullable=False)

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def initialize_database():
    # 데이터베이스 초기화: 기존 테이블 삭제 후 다시 생성
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

def insert_fake_vectors(session, num_vectors=100000, dim=1000):
    times = []
    for i in range(num_vectors):
        fake_vector = np.random.rand(dim).tolist()
        start_time = time.time()
        image_embedding = ImageEmbedding(
            image_path=f'fake_path_{i}',
            label='fake_label',
            embedding=fake_vector,
            embedding_vector=fake_vector
        )
        session.add(image_embedding)
        session.commit()
        end_time = time.time()
        times.append(end_time - start_time)
        if (i + 1) % 10000 == 0:
            print(f"Insertion - {i + 1} vectors inserted")
            avg_time = mean(times[-10000:])
            stddev_time = stdev(times[-10000:])
            print(f"Average Time: {avg_time:.6f} seconds, Standard Deviation: {stddev_time:.6f} seconds")
    return times

def build_hnsw_index(session, dim=1000):
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=100000, ef_construction=200, M=16)
    
    embeddings = []
    ids = []

    results = session.query(ImageEmbedding).all()
    for result in results:
        embeddings.append(result.embedding_vector)
        ids.append(result.id)

    p.add_items(embeddings, ids)

    index_data = pickle.dumps(p)
    hnsw_index = HNSWIndex(index_data=index_data)
    session.add(hnsw_index)
    session.commit()

if __name__ == "__main__":
    initialize_database()
    times = insert_fake_vectors(session)
    avg_time = mean(times)
    stddev_time = stdev(times)
    print(f"Insertion - Total Average Time: {avg_time:.6f} seconds, Standard Deviation: {stddev_time:.6f} seconds")

    build_hnsw_index(session)
