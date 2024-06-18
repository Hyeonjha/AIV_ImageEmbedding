import time
import numpy as np
from sqlalchemy import create_engine, text
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

def find_similar_images(input_image_path, session, threshold=0.8, top_n=5):
    img = Image.open(input_image_path).convert('RGB')
    img2vec = Img2Vec()
    input_embedding = img2vec.get_vec(img).tolist()
    
    # Postgres의 pgvector 사용하여 유사도 계산 및 검색
    query = text(f"""
        SELECT image_path, label, 1 - (embedding_vector <=> '[{','.join(map(str, input_embedding))}]') AS similarity
        FROM image_embeddings
        WHERE embedding_vector <=> '[{','.join(map(str, input_embedding))}]' <= {1 - threshold}
        ORDER BY embedding_vector <=> '[{','.join(map(str, input_embedding))}]'
        LIMIT :top_n;
    """)

    results = session.execute(query, {'top_n': top_n}).fetchall()
    similarities = [(row.image_path, row.label, row.similarity) for row in results]

    return similarities

def measure_search_time(folder_path, session, threshold=0.8, top_n=5, repeat=20):  # 10000
    search_times = []
    images = []
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            images.append(img_path)
    
    for i in range(repeat):
        img_path = images[i % len(images)]
        start_time = time.time()
        find_similar_images(img_path, session, threshold, top_n)
        end_time = time.time()
        search_times.append(end_time - start_time)
    
    search_times_np = np.array(search_times)
    print(f"Search times over {repeat} iterations: {search_times_np.mean()} ± {search_times_np.std()} seconds")
    return search_times_np.mean(), search_times_np.std()

if __name__ == "__main__":
    folder_path = 'data-gatter/test'
    measure_search_time(folder_path, session, repeat=20)  # 10000
