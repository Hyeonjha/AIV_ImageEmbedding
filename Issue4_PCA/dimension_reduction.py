# dimension_reduction.py
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.preprocessing import LabelEncoder, StandardScaler
import umap
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import tensorflow as tf
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import KernelPCA


# n_components -> 보통 2 또는 3으로 설정하여 2D/3D로 시각화

### PCA
def reduce_dimensions_pca(embeddings, n_components=3):  
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

### t-SNE
# perplexity : 데이터 포인트 주변의 지역 밀도를 조절하는 값. (보통 5 ~ 50. 커지면 더 큰 데이터 세트에 대해 잘 동작, 작으면 지역구조 더 잘 반영 but 노이즈에 더 민감) 
# n_iter : 최적화를 위한 반복 횟수 (보통 300 ~ 1000) - 더 많은 반복 통해 최적화 but 계산 시간 길어짐
def reduce_dimensions_tsne(embeddings, n_components=3, perplexity=10, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

### UMAP
# n_neighbors : 각 데이터 포인트에 대한 근접 이웃의 수 (클수록 전반적인 구조 잘 반영, 작을수록 국부적 구조 잘 반영) 15
# min_dist : 임베딩 공간에서 최소 거리. 클러스터의 밀도에 영향을 미침 (작을수록 클러스터가 더 조밀하게 모이고, 클수록 분산) 0.1
def reduce_dimensions_umap(embeddings, n_components=3,  n_neighbors=20, min_dist=0.4):  ### n_neighbors -> # of class 
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

### MDS
def reduce_dimensions_mds(embeddings, n_components=3):
    mds = MDS(n_components=n_components, random_state=42)
    reduced_embeddings = mds.fit_transform(embeddings)
    return reduced_embeddings

### Isomap
# n_neighbors : 각 데이터 포인트에 대한 근접 이웃의 수 
def reduce_dimensions_isomap(embeddings, n_components=3, n_neighbors=12):
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    reduced_embeddings = isomap.fit_transform(embeddings)
    return reduced_embeddings

### LDA
def reduce_dimensions_lda(embeddings, labels, n_components=2):
    lda = LDA(n_components=n_components)
    reduced_embeddings = lda.fit_transform(embeddings, labels)
    return reduced_embeddings

### Autoencoder
def reduce_dimensions_autoencoder(embeddings, encoding_dim=2, epochs=50, batch_size=256):
    input_dim = embeddings.shape[1]
    input_img = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_img)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = tf.keras.models.Model(input_img, decoded)
    encoder = tf.keras.models.Model(input_img, encoded)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(embeddings, embeddings, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
    
    reduced_embeddings = encoder.predict(embeddings)
    return reduced_embeddings

### Factor Analysis
def reduce_dimensions_fa(embeddings, n_components=2):
    fa = FactorAnalysis(n_components=n_components)
    reduced_embeddings = fa.fit_transform(embeddings)
    return reduced_embeddings

### Kernel PCA
def reduce_dimensions_kernel_pca(embeddings, n_components=2, kernel='rbf'):
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    reduced_embeddings = kpca.fit_transform(embeddings)
    return reduced_embeddings

def normalize_embeddings(embeddings):
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    return normalized_embeddings

def plot_embeddings_2d(embeddings, labels, title='Embedding Visualization'):
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=numeric_labels, cmap='viridis')
    plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)))
    plt.clim(-0.5, len(label_encoder.classes_)-0.5)
    plt.title(title)
    plt.show()

def plot_embeddings_3d(embeddings, labels, title='Embedding Visualization'):
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=numeric_labels, cmap='viridis')
    fig.colorbar(scatter, ticks=range(len(label_encoder.classes_)))
    plt.title(title)
    plt.show()
