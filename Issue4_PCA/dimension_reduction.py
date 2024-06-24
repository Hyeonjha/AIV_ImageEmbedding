# dimension_reduction.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.preprocessing import LabelEncoder, StandardScaler
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reduce_dimensions_pca(embeddings, n_components=3):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def reduce_dimensions_tsne(embeddings, n_components=3, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def reduce_dimensions_umap(embeddings, n_components=3, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def reduce_dimensions_mds(embeddings, n_components=3):
    mds = MDS(n_components=n_components, random_state=42)
    reduced_embeddings = mds.fit_transform(embeddings)
    return reduced_embeddings

def reduce_dimensions_isomap(embeddings, n_components=3, n_neighbors=5):
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    reduced_embeddings = isomap.fit_transform(embeddings)
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
