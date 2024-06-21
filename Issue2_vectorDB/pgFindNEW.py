import numpy as np
import pickle
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from Img2Vec import Img2Vec
from ImageEmbedding import ImageEmbedding, HNSWIndex, Base  # 모델 정의를 포함한 모듈에서 가져옴
from sqlalchemy.ext.declarative import declarative_base

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

def find_similar_images(input_image_path, session, hnsw_index, top_n=5):
    img = Image.open(input_image_path).convert('RGB')
    img2vec = Img2Vec()
    input_embedding = img2vec.get_vec(img).tolist()

    labels, distances = hnsw_index.knn_query(np.array(input_embedding), k=top_n)
    nearest_ids = labels[0].tolist()

    query = session.query(ImageEmbedding).filter(ImageEmbedding.id.in_(nearest_ids))
    results = query.all()
    similarities = [(res.image_path, res.label, 1 - distances[0][i]) for i, res in enumerate(results)]
    return similarities

if __name__ == "__main__":
    hnsw_index = load_hnsw_index(session)

    input_image_path = 'data-gatter/test/bubble_380033.jpg'
    similar_images = find_similar_images(input_image_path, session, hnsw_index)
    for image_path, label, cosine_similarity in similar_images:
        print(f"Image Path: {image_path}, Label: {label}, Similarity: {cosine_similarity:.4f}")
