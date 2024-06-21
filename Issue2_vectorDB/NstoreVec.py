import os
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from Img2Vec import Img2Vec
from ImageEmbedding import ImageEmbedding, Base

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

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
                    embedding_vector=embedding  # pgvector 저장
                )
                session.add(image_embedding)
    session.commit()

if __name__ == "__main__":
    folder_path = 'data-gatter/train_L'
    initialize_database()  # 데이터베이스 초기화
    save_image_embedding(folder_path, session)
