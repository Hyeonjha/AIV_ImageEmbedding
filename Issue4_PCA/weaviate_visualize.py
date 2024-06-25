# weaviate_visualize.py
import sys
import os
import time

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
    reduce_dimensions_lda,
    reduce_dimensions_autoencoder,
    reduce_dimensions_fa,
    reduce_dimensions_kernel_pca,
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
    
    # 2D Visualizations with Time Measurement
    def measure_time(func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    ### 2D Visualizations with Time Measurement
    # PCA 차원 축소 및 시각화
    pca_embeddings, pca_time = measure_time(reduce_dimensions_pca, normalized_embeddings, n_components=2)
    plot_embeddings_2d(pca_embeddings, labels, title='PCA 2D Visualization')
    print(f"PCA 2D reduction time: {pca_time:.4f} seconds")

    # t-SNE 차원 축소 및 시각화
    tsne_embeddings, tsne_time = measure_time(reduce_dimensions_tsne, normalized_embeddings, n_components=2)
    plot_embeddings_2d(tsne_embeddings, labels, title='t-SNE 2D Visualization')
    print(f"t-SNE 2D reduction time: {tsne_time:.4f} seconds")

    # UMAP 차원 축소 및 시각화
    umap_embeddings, umap_time = measure_time(reduce_dimensions_umap, normalized_embeddings, n_components=2)
    plot_embeddings_2d(umap_embeddings, labels, title='UMAP 2D Visualization')
    print(f"UMAP 2D reduction time: {umap_time:.4f} seconds")

    # MDS 차원 축소 및 시각화
    mds_embeddings, mds_time = measure_time(reduce_dimensions_mds, normalized_embeddings, n_components=2)
    plot_embeddings_2d(mds_embeddings, labels, title='MDS 2D Visualization')
    print(f"MDS 2D reduction time: {mds_time:.4f} seconds")

    # Isomap 차원 축소 및 시각화
    isomap_embeddings, isomap_time = measure_time(reduce_dimensions_isomap, normalized_embeddings, n_components=2)
    plot_embeddings_2d(isomap_embeddings, labels, title='Isomap 2D Visualization')
    print(f"Isomap 2D reduction time: {isomap_time:.4f} seconds")

    # LDA 차원 축소 및 시각화
    lda_embeddings, lda_time = measure_time(reduce_dimensions_lda, normalized_embeddings, labels, n_components=2)
    plot_embeddings_2d(lda_embeddings, labels, title='LDA 2D Visualization')
    print(f"LDA 2D reduction time: {lda_time:.4f} seconds")

    # Autoencoders 차원 축소 및 시각화
    autoencoder_embeddings, autoencoder_time = measure_time(reduce_dimensions_autoencoder, normalized_embeddings, encoding_dim=2)
    plot_embeddings_2d(autoencoder_embeddings, labels, title='Autoencoders 2D Visualization')
    print(f"Autoencoders 2D reduction time: {autoencoder_time:.4f} seconds")

    # Factor Analysis 차원 축소 및 시각화
    fa_embeddings, fa_time = measure_time(reduce_dimensions_fa, normalized_embeddings, n_components=2)
    plot_embeddings_2d(fa_embeddings, labels, title='Factor Analysis 2D Visualization')
    print(f"Factor Analysis 2D reduction time: {fa_time:.4f} seconds")

    # Kernel PCA 차원 축소 및 시각화
    kernel_pca_embeddings, kernel_pca_time = measure_time(reduce_dimensions_kernel_pca, normalized_embeddings, n_components=2)
    plot_embeddings_2d(kernel_pca_embeddings, labels, title='Kernel PCA 2D Visualization')
    print(f"Kernel PCA 2D reduction time: {kernel_pca_time:.4f} seconds")

    ### 3D Visualizations with Time Measurement
    # PCA 차원 축소 및 시각화
    pca_embeddings_3d, pca_time_3d = measure_time(reduce_dimensions_pca, normalized_embeddings, n_components=3)
    plot_embeddings_3d(pca_embeddings_3d, labels, title='PCA 3D Visualization')
    print(f"PCA 3D reduction time: {pca_time_3d:.4f} seconds")

    # t-SNE 차원 축소 및 시각화
    tsne_embeddings_3d, tsne_time_3d = measure_time(reduce_dimensions_tsne, normalized_embeddings, n_components=3)
    plot_embeddings_3d(tsne_embeddings_3d, labels, title='t-SNE 3D Visualization')
    print(f"t-SNE 3D reduction time: {tsne_time_3d:.4f} seconds")

    # UMAP 차원 축소 및 시각화
    umap_embeddings_3d, umap_time_3d = measure_time(reduce_dimensions_umap, normalized_embeddings, n_components=3)
    plot_embeddings_3d(umap_embeddings_3d, labels, title='UMAP 3D Visualization')
    print(f"UMAP 3D reduction time: {umap_time_3d:.4f} seconds")

    # MDS 차원 축소 및 시각화
    mds_embeddings_3d, mds_time_3d = measure_time(reduce_dimensions_mds, normalized_embeddings, n_components=3)
    plot_embeddings_3d(mds_embeddings_3d, labels, title='MDS 3D Visualization')
    print(f"MDS 3D reduction time: {mds_time_3d:.4f} seconds")

    # Isomap 차원 축소 및 시각화
    isomap_embeddings_3d, isomap_time_3d = measure_time(reduce_dimensions_isomap, normalized_embeddings, n_components=3)
    plot_embeddings_3d(isomap_embeddings_3d, labels, title='Isomap 3D Visualization')
    print(f"Isomap 3D reduction time: {isomap_time_3d:.4f} seconds")

    # LDA 차원 축소 및 시각화
    lda_embeddings_3d, lda_time_3d = measure_time(reduce_dimensions_lda, normalized_embeddings, labels, n_components=3)
    plot_embeddings_3d(lda_embeddings_3d, labels, title='LDA 3D Visualization')
    print(f"LDA 3D reduction time: {lda_time_3d:.4f} seconds")

    # Autoencoders 차원 축소 및 시각화
    autoencoder_embeddings_3d, autoencoder_time_3d = measure_time(reduce_dimensions_autoencoder, normalized_embeddings, encoding_dim=3)
    plot_embeddings_3d(autoencoder_embeddings_3d, labels, title='Autoencoders 3D Visualization')
    print(f"Autoencoders 3D reduction time: {autoencoder_time_3d:.4f} seconds")

    # Factor Analysis 차원 축소 및 시각화
    fa_embeddings_3d, fa_time_3d = measure_time(reduce_dimensions_fa, normalized_embeddings, n_components=3)
    plot_embeddings_3d(fa_embeddings_3d, labels, title='Factor Analysis 3D Visualization')
    print(f"Factor Analysis 3D reduction time: {fa_time_3d:.4f} seconds")

    # Kernel PCA 차원 축소 및 시각화
    kernel_pca_embeddings_3d, kernel_pca_time_3d = measure_time(reduce_dimensions_kernel_pca, normalized_embeddings, n_components=3)
    plot_embeddings_3d(kernel_pca_embeddings_3d, labels, title='Kernel PCA 3D Visualization')
    print(f"Kernel PCA 3D reduction time: {kernel_pca_time_3d:.4f} seconds")