# weaviate_visualize.py
import sys
import os

# 필요한 모듈 경로를 추가
issue2_openDB_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Issue2_openDB'))
sys.path.append(issue2_openDB_path)

import weaviate
import numpy as np
from dimension_reduction import (
    reduce_dimensions_pca,
    reduce_dimensions_tsne,
    reduce_dimensions_umap,
    reduce_dimensions_mds,
    reduce_dimensions_isomap,
    normalize_embeddings,
    plot_embeddings_2d,
    plot_embeddings_3d
)
from sklearn.preprocessing import StandardScaler

# Weaviate 클라이언트 설정
client = weaviate.Client(url="http://localhost:8080")

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
    normalized_embeddings = normalize_embeddings(embeddings)
    
    # 2D Visualizations
    # PCA 차원 축소 및 시각화
    pca_embeddings = reduce_dimensions_pca(normalized_embeddings, n_components=2)
    plot_embeddings_2d(pca_embeddings, labels, title='PCA 2D Visualization')

    # t-SNE 차원 축소 및 시각화
    tsne_embeddings = reduce_dimensions_tsne(normalized_embeddings, n_components=2)
    plot_embeddings_2d(tsne_embeddings, labels, title='t-SNE 2D Visualization')

    # UMAP 차원 축소 및 시각화
    umap_embeddings = reduce_dimensions_umap(normalized_embeddings, n_components=2)
    plot_embeddings_2d(umap_embeddings, labels, title='UMAP 2D Visualization')

    # MDS 차원 축소 및 시각화
    mds_embeddings = reduce_dimensions_mds(normalized_embeddings, n_components=2)
    plot_embeddings_2d(mds_embeddings, labels, title='MDS 2D Visualization')

    # Isomap 차원 축소 및 시각화
    isomap_embeddings = reduce_dimensions_isomap(normalized_embeddings, n_components=2)
    plot_embeddings_2d(isomap_embeddings, labels, title='Isomap 2D Visualization')
    
    # 3D Visualizations
    # PCA 차원 축소 및 시각화
    pca_embeddings_3d = reduce_dimensions_pca(normalized_embeddings, n_components=3)
    plot_embeddings_3d(pca_embeddings_3d, labels, title='PCA 3D Visualization')

    # t-SNE 차원 축소 및 시각화
    tsne_embeddings_3d = reduce_dimensions_tsne(normalized_embeddings, n_components=3)
    plot_embeddings_3d(tsne_embeddings_3d, labels, title='t-SNE 3D Visualization')

    # UMAP 차원 축소 및 시각화
    umap_embeddings_3d = reduce_dimensions_umap(normalized_embeddings, n_components=3)
    plot_embeddings_3d(umap_embeddings_3d, labels, title='UMAP 3D Visualization')

    # MDS 차원 축소 및 시각화
    mds_embeddings_3d = reduce_dimensions_mds(normalized_embeddings, n_components=3)
    plot_embeddings_3d(mds_embeddings_3d, labels, title='MDS 3D Visualization')

    # Isomap 차원 축소 및 시각화
    isomap_embeddings_3d = reduce_dimensions_isomap(normalized_embeddings, n_components=3)
    plot_embeddings_3d(isomap_embeddings_3d, labels, title='Isomap 3D Visualization')
