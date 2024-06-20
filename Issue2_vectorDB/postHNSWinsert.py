import os
import hnswlib
import numpy as np
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from Img2Vec import Img2Vec
from ImageEmbedding import ImageEmbedding, Base
import psycopg2

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def initialize_database():
    # 데이터베이스 초기화: 기존 테이블 삭제 후 다시 생성
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

def save_image_embedding(folder_path, session):
    img2vec = Img2Vec()
    dim = 1000  # 벡터 차원 수
    index = hnswlib.Index(space='cosine', dim=dim)
    id_counter = 0
    embeddings = []
    ids = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                img = Image.open(img_path).convert('RGB')
                embedding = img2vec.get_vec(img)
                embeddings.append(embedding)
                ids.append(id_counter)
                image_embedding = ImageEmbedding(
                    image_path=img_path,
                    label=label,
                    embedding=embedding.tolist(),
                    embedding_vector=embedding  # pgvector 저장
                )
                session.add(image_embedding)
                id_counter += 1

    session.commit()

    embeddings = np.array(embeddings)
    index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
    index.add_items(embeddings, ids)
    index.save_index("hnsw_index.bin")

if __name__ == "__main__":
    folder_path = 'data-gatter/train_L'
    initialize_database()  # 데이터베이스 초기화
    save_image_embedding(folder_path, session)
