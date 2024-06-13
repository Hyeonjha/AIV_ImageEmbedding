import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
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

def save_image_embedding(folder_path, session):
    img2vec = Img2Vec()
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                img = Image.open(img_path).convert('RGB')
                embedding = img2vec.get_vec(img)
                image_embedding = ImageEmbedding(image_path=img_path, label=label, embedding=embedding.tolist())
                session.add(image_embedding)
    session.commit()

if __name__ == "__main__":
    folder_path = 'data-gatter/train_L'
    save_image_embedding(folder_path, session)
