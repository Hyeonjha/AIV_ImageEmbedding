import os
import numpy as np
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datasketch import MinHash, MinHashLSH
from Img2Vec import Img2Vec
from ImageEmbedding import ImageEmbedding, Base

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def convert_embedding_to_minhash(embedding, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for value in embedding:
        m.update(str(value).encode('utf8'))
    return m

def find_similar_lsh(lsh, query_embedding, num_perm=128):
    m = convert_embedding_to_minhash(query_embedding, num_perm)
    print(f"MinHash for query: {m.hashvalues[:10]}")  # 디버깅용 출력
    print(f"lsh.q : {lsh.query(m)}") # 여기부터 출력 안됨 ->>>> lsh.q : []
    return lsh.query(m)

def find_similar_images(input_image_path, session, lsh, num_perm=128, top_n=5):
    print("Start finding similar images")
    img = Image.open(input_image_path).convert('RGB')
    img2vec = Img2Vec()
    input_embedding = img2vec.get_vec(img).tolist()
    
    print(f"Input embedding: {input_embedding[:10]}")  # 입력 벡터의 일부를 출력
    
    lsh_results = find_similar_lsh(lsh, input_embedding, num_perm=num_perm)
    results = []
    print(f"LSH results: {lsh_results}")    # LSH results: []
    if lsh_results:
        lsh_results_placeholders = ', '.join([':result' + str(i) for i in range(len(lsh_results))])
        query = text(f"""
            SELECT image_path, label, embedding, 1 - (embedding_vector <=> '[{','.join(map(str, input_embedding))}]'::vector) AS similarity
            FROM image_embeddings
            WHERE image_path IN ({lsh_results_placeholders})
            ORDER BY similarity DESC
            LIMIT :top_n;
        """)
        params = {'top_n': top_n}
        for i, result in enumerate(lsh_results):
            params['result' + str(i)] = result

        print("Before executing DB query")
        db_results = session.execute(query, params).fetchall()
        print("After executing DB query")
        if db_results:
            for db_result in db_results:
                results.append((db_result.image_path, db_result.label, db_result.similarity))
                print(f"Query Result: {db_result.image_path}, Similarity: {db_result.similarity:.4f}")  # 쿼리 결과를 출력하여 확인
        else:
            print("No DB results found")

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]

if __name__ == "__main__":
    input_image_path = 'data-gatter/test/bubble_380033.jpg'
    
    lsh = MinHashLSH(threshold=0.8, num_perm=128)  # 임계값 조정
    
    # Load existing embeddings into LSH
    embeddings = session.query(ImageEmbedding).all()
    print(f"Number of embeddings loaded: {len(embeddings)}")
    for embedding in embeddings:
        m = convert_embedding_to_minhash(embedding.embedding, num_perm=128)
        lsh.insert(embedding.image_path, m)
        #print(f"Inserted {embedding.image_path} into LSH")

    # Print all keys in LSH
    #print(f"Keys in LSH: {list(lsh.keys)}")
    
    similar_images = find_similar_images(input_image_path, session, lsh, num_perm=128, top_n=5)
    print("End finding")

    for image_path, label, similarity in similar_images:
        print(f"Image Path: {image_path}, Label: {label}, Similarity: {similarity:.4f}")
