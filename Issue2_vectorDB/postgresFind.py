import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.metrics.pairwise import cosine_similarity
import timm

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
Base = declarative_base()

class ImageEmbedding(Base):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

class Img2Vec():
    def __init__(self, model_name='convnext_base', cuda=False):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        self.model.eval()

    def get_vec(self, img):
        image = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image)
        embedding_np = embedding.cpu().numpy().flatten()
        return embedding_np

def find_similar_images(input_image_path, session, threshold=0.8, top_n=5):
    img = Image.open(input_image_path).convert('RGB')
    img2vec = Img2Vec()
    input_embedding = img2vec.get_vec(img).tolist()
    
    embeddings = session.query(ImageEmbedding).all()
    similarities = []

    for image_embedding in embeddings:
        stored_embedding = np.array(image_embedding.embedding)
        similarity = cosine_similarity([input_embedding], [stored_embedding])[0][0]
        if similarity >= threshold:
            similarities.append((image_embedding.image_path, image_embedding.label, similarity))

    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_n]

if __name__ == "__main__":
    input_image_path = 'data-gatter/testcopy/bubble_381007.jpg'
    similar_images = find_similar_images(input_image_path, session)
    print(similar_images)
