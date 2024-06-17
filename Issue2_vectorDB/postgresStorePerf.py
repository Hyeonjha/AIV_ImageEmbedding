import time
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from PIL import Image
import os
from Img2Vec import Img2Vec
from ImageEmbedding import ImageEmbedding, Base

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

def save_image_embedding(folder_path, session, repeat=20):  # 10000
    img2vec = Img2Vec()
    insert_times = []
    images = []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                images.append((img_path, label))
    
    for i in range(repeat):
        img_path, label = images[i % len(images)]
        if session.query(ImageEmbedding).filter_by(image_path=img_path).first():
            continue  # 중복된 이미지 경로가 있을 경우 건너뜁니다.
        
        img = Image.open(img_path).convert('RGB')
        embedding = img2vec.get_vec(img)
        image_embedding = ImageEmbedding(
            image_path=img_path,
            label=label,
            embedding=embedding.tolist(),
            embedding_vector=embedding  # pgvector 저장
        )
        start_time = time.time()
        session.add(image_embedding)
        session.commit()
        end_time = time.time()
        insert_times.append(end_time - start_time)
    
    insert_times_np = np.array(insert_times)
    print(f"Insert times over {repeat} iterations: {insert_times_np.mean()} ± {insert_times_np.std()} seconds")
    return insert_times_np.mean(), insert_times_np.std()

if __name__ == "__main__":
    folder_path = 'data-gatter/train_L'

    initialize_database(session)  # 데이터베이스 초기화
    save_image_embedding(folder_path, session, repeat=20)  # 10000
