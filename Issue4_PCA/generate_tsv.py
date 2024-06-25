# generate_tsv.py
import sys
import os
import weaviate
import numpy as np
import pandas as pd

# Weaviate 클라이언트 설정
client = weaviate.Client("http://localhost:8080")

# 임베딩 데이터 가져오기
def fetch_all_embeddings(client, class_name='ImageEmbedding', batch_size=100):
    embeddings = []
    labels = []
    offset = 0

    while True:
        query_result = client.query.get(class_name, ["image_path", "label", "_additional {vector}"])\
            .with_limit(batch_size)\
            .with_offset(offset)\
            .do()

        data_batch = query_result['data']['Get'][class_name]
        if not data_batch:
            break

        for item in data_batch:
            embeddings.append(item['_additional']['vector'])
            labels.append(item['label'])

        offset += batch_size

    return np.array(embeddings), labels

if __name__ == "__main__":
    embeddings, labels = fetch_all_embeddings(client)
    
    # 임베딩 데이터를 tsv 파일로 저장
    pd.DataFrame(embeddings).to_csv("embeddings_tensor.tsv", sep='\t', header=False, index=False)
    pd.DataFrame(labels).to_csv("embeddings_metadata.tsv", sep='\t', header=False, index=False)
