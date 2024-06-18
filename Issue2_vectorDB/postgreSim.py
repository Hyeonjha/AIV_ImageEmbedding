import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
        SELECT image_path, label, embedding, 1 - (embedding_vector <=> '[{','.join(map(str, input_embedding))}]') AS cosine_similarity
        FROM image_embeddings
        WHERE embedding_vector <=> '[{','.join(map(str, input_embedding))}]' <= {1 - threshold}
        ORDER BY embedding_vector <=> '[{','.join(map(str, input_embedding))}]'
        LIMIT :top_n;
    """)

    results = session.execute(query, {'top_n': top_n}).fetchall()
    similarities = [(row.image_path, row.label, row.embedding, row.similarity) for row in results]

    return similarities

def compare_similarities(folder_path, session, threshold=0.8, top_n=5):
    discrepancies = []
    img2vec = Img2Vec()
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            postgres_results = find_similar_images(img_path, session, threshold, top_n)
            
            img = Image.open(img_path).convert('RGB')
            input_embedding = img2vec.get_vec(img)
            
            for result in postgres_results:
                stored_embedding = np.array(result[2])
                calculated_similarity = cosine_similarity([input_embedding], [stored_embedding])[0][0]
                
                if not np.isclose(calculated_similarity, result[3], atol=1e-6):
                    discrepancies.append((result[0], result[1], calculated_similarity, result[3]))
    
    if discrepancies:
        print("Discrepancies found between PostgreSQL and manually calculated similarities:")
        for discrepancy in discrepancies:
            print(f"Image: {discrepancy[0]}, Label: {discrepancy[1]}, Calculated: {discrepancy[2]}, PostgreSQL: {discrepancy[3]}")
    else:
        print("All similarities match between PostgreSQL and manually calculated values.")

if __name__ == "__main__":
    folder_path = 'data-gatter/testcopy'
    compare_similarities(folder_path, session)
