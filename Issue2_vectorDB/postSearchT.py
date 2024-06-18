import time
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from ImageEmbedding import ImageEmbedding, Base

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def find_similar_embeddings(random_embedding, session, threshold=0.8, top_n=5):
    query = text(f"""
        SELECT image_path, label, 1 - (embedding_vector <=> '[{','.join(map(str, random_embedding))}]') AS similarity
        FROM image_embeddings
        WHERE embedding_vector <=> '[{','.join(map(str, random_embedding))}]' <= {1 - threshold}
        ORDER BY embedding_vector <=> '[{','.join(map(str, random_embedding))}]'
        LIMIT :top_n;
    """)

    results = session.execute(query, {'top_n': top_n}).fetchall()
    similarities = [(row.image_path, row.label, row.similarity) for row in results]

    return similarities

def measure_search_time(session, threshold=0.8, top_n=5, total_tests=100000, dim=1000):
    search_times = []
    random_embeddings = [np.random.rand(dim).tolist() for _ in range(total_tests)]

    for idx, random_embedding in enumerate(random_embeddings):
        start_time = time.time()
        find_similar_embeddings(random_embedding, session, threshold, top_n)
        end_time = time.time()
        search_times.append(end_time - start_time)
        
        if (idx + 1) % 1000 == 0:
            search_times_np = np.array(search_times)
            print(f"Search times for {idx + 1} queries: {search_times_np.mean()} ± {search_times_np.std()} seconds")
            search_times = []  # Reset for the next batch

    return search_times_np.mean(), search_times_np.std()

if __name__ == "__main__":
    measure_search_time(session, total_tests=100000)
