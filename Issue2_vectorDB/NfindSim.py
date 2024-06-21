from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from PIL import Image
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

    # 디버깅 출력 추가
    #print(f"Input Embedding: {input_embedding}")
    
    # Postgres의 pgvector 사용하여 유사도 계산 및 검색
    query = text(f"""
        SELECT image_path, label, 1 - (embedding_vector <=> '{input_embedding}'::vector) AS cosine_similarity
        FROM image_embeddings
        WHERE embedding_vector <=> '{input_embedding}'::vector <= {1 - threshold}
        ORDER BY embedding_vector <=> '{input_embedding}'::vector
        LIMIT :top_n;
    """)

    # 디버깅 출력 추가
    #print(f"Query: {query}")

    results = session.execute(query, {
        'embedding': input_embedding,
        'threshold': 1 - threshold,
        'top_n': top_n
    }).fetchall()

    similarities = [(row.image_path, row.label, row.cosine_similarity) for row in results]

    return similarities

if __name__ == "__main__":
    input_image_path = 'data-gatter/test/react_380116.jpg'
    similar_images = find_similar_images(input_image_path, session)
    print(similar_images)
    for image_path, label, cosine_similarity in similar_images:
        print(f"Image Path: {image_path}, Label: {label}, Similarity: {cosine_similarity:.4f}")
