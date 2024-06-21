import os
import hnswlib 
import pickle
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from Img2Vec import Img2Vec

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

def save_image_embedding(folder_path, session):
    img2vec = Img2Vec()
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                img = Image.open(img_path).convert('RGB')
                embedding = img2vec.get_vec(img)
                image_embedding = ImageEmbedding(
                    image_path=img_path,
                    label=label,
                    embedding=embedding.tolist(),
                    embedding_vector=embedding.tolist()  # pgvector 저장
                )
                session.add(image_embedding)
    session.commit()

def build_hnsw_index(session, dim=1000):
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=10000, ef_construction=200, M=16)
    
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
    folder_path = 'data-gatter/train_L'
    initialize_database()  # 데이터베이스 초기화
    save_image_embedding(folder_path, session)
    
    # HNSW 인덱스를 생성하여 데이터베이스에 저장
    build_hnsw_index(session)
