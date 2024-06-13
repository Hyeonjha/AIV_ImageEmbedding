import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import os
import ssl
import matplotlib.pyplot as plt
from open_clip import create_model_and_transforms
import timm  # Import timm for additional models
import torchvision.models as models

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
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        try:
            if model_name.startswith("coca_"):
                self.model, _, self.transform = create_model_and_transforms(model_name, pretrained='mscoco_finetuned_laion2b_s13b_b90k')
            elif hasattr(models, model_name):
                self.model = getattr(models, model_name)(pretrained=True).to(self.device)
            else:
                self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        except AttributeError:
            raise ValueError(f"Model {model_name} not available in torchvision.models or timm")
        except RuntimeError as e:
            if "No pretrained weights exist" in str(e):
                self.model = timm.create_model(model_name, pretrained=False).to(self.device)
            else:
                raise e
        self.model.eval()

    def get_vec(self, img):
        if self.model is None:
            return None
        image = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image)
        embedding_np = embedding.cpu().numpy().flatten()
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

def classify_images(model_dict, folder_path, cuda=False):
    images, labels = load_images_from_folder(folder_path)
    results = {}

    for model_display_name, model_name in model_dict.items():
        print(f"Evaluating model {model_name}")
        if model_name.startswith("coca_"):
            img2vec = CoCaImg2Vec(model_name, pretrained='mscoco_finetuned_laion2b_s13b_b90k', cuda=cuda)
        else:
            img2vec = Img2Vec(model_name, cuda=cuda)
        if img2vec.model is None:
            continue

        start_time = time.time()
        embeddings = [img2vec.get_vec(img) for img in images]
        embeddings = [e for e in embeddings if e is not None]
        embeddings = np.array(embeddings)
        processing_time = (time.time() - start_time) / len(images)

        # 벡터 차원 확인
        vector_dimension = embeddings.shape[1] if embeddings.size > 0 else "Unknown"

        results[model_display_name] = {
            'processing_time': processing_time,
            'vector_dimension': vector_dimension
        }

    return results

if __name__ == "__main__":
    folder_path = './data-gatter/train_L'

    model_dict = {
        'OmniVec (ViT)': 'vit_b_16',
        'CoCa': 'coca_ViT-B-32',
        'Swin Transformer V2': 'swin_v2_b',
        'ConvNeXt': 'convnext_base',
        'EfficientNetV2': 'efficientnet_v2_s',
        'RegNet': 'regnet_y_16gf',
        'DeiT': 'deit_base_patch16_224',
        'NFNet': 'nfnet_f0',
        'ResNet18': 'resnet18',
        'ResNet50': 'resnet50',
        'EfficientNet B0': 'efficientnet_b0',
        'EfficientNet B7': 'efficientnet_b7'
    }

    cuda = torch.cuda.is_available()

    results = classify_images(model_dict, folder_path, cuda)
    for model_name, result in results.items():
        print(f"\nModel: {model_name}")
        print(f"Processing Time per Image: {result['processing_time']} seconds")
        print(f"Vector Dimension: {result['vector_dimension']}")
