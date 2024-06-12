import os
import ssl
import time
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pinecone import Pinecone, ServerlessSpec
from open_clip import create_model_and_transforms
from torchvision import transforms, models

# SSL 인증서 설정
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

class CoCaImg2Vec():
    def __init__(self, model_name, pretrained, cuda=False):
        self.model, _, self.transform = create_model_and_transforms(model_name, pretrained=pretrained)
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_vec(self, img):
        image = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image).cpu().numpy().flatten()
        return embedding

class Img2Vec():
    def __init__(self, model_name, cuda=False):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = getattr(models, model_name)(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_vec(self, img):
        image = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image)
        embedding_np = embedding.cpu().numpy().flatten()
        #print(f"Embedding dimension for {self.model.__class__.__name__}: {embedding_np.shape[0]}")
        return embedding_np

def load_images_from_folder(folder):
    images = []
    labels = []
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for class_folder_name in os.listdir(folder):
        class_folder_path = os.path.join(folder, class_folder_name)
        if not os.path.isdir(class_folder_path):
            continue
        for filename in os.listdir(class_folder_path):
            img_path = os.path.join(class_folder_path, filename)
            if os.path.splitext(filename)[1].lower() in valid_image_extensions:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                labels.append(class_folder_name)
    return images, labels

def create_pinecone_index(api_key, index_name, dimension):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region="us-east-1"
            )
        )
    return pc.Index(index_name)

def save_embeddings_to_pinecone(index, embeddings, labels, namespace):
    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            "id": f"vec{i+1}",
            "values": emb.tolist(),
            "metadata": {"label": labels[i]}
        })

    index.upsert(
        vectors=vectors,
        namespace=namespace
    )
    
    return vectors

def classify_images_with_pinecone(model_dict, folder_path, api_key, index_name, namespace, cuda=False):
    images, labels = load_images_from_folder(folder_path)
    all_vectors = []

    for model_name, model in model_dict.items():
        print(f"Evaluating model {model_name}")
        img2vec = Img2Vec(model, cuda=cuda)
        if img2vec.model is None:
            continue

        start_time = time.time()
        embeddings = [img2vec.get_vec(img) for img in images]
        embeddings = [e for e in embeddings if e is not None]
        embeddings = np.array(embeddings)
        processing_time = (time.time() - start_time) / len(images)
        print(f"Processing Time per Image: {processing_time}")

        # Create Pinecone index
        index = create_pinecone_index(api_key, index_name, embeddings.shape[1])
        
        # Save embeddings to Pinecone
        vectors = save_embeddings_to_pinecone(index, embeddings, labels, namespace)
        all_vectors.extend(vectors)

    return all_vectors

def visualize_embeddings(vectors):
    print("Embedding vectors and their metadata:")
    for vector in vectors:
        print(f"ID: {vector['id']}, Label: {vector['metadata']['label']}, Vector: {vector['values'][:5]}...")

if __name__ == "__main__":
    folder_path = './data-gatter/train_L'
    api_key = 'f5001027-9bd4-4abb-8dd6-a2db16540ecc'
    index_name = 'convnext'
    namespace = 'ns1'
    
    model_dict = {
        'ConvNeXt': 'convnext_base',
        #'EfficientNetV2': 'efficientnet_v2_s'
    }

    cuda = torch.cuda.is_available()

    vectors = classify_images_with_pinecone(model_dict, folder_path, api_key, index_name, namespace, cuda)
    visualize_embeddings(vectors)
