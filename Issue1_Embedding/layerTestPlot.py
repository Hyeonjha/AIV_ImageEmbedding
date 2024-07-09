import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import ssl
import matplotlib.pyplot as plt
import seaborn as sns
from open_clip import create_model_and_transforms
import timm  # Import timm for additional models
import torchvision.models as models

# SSL 인증서 설정
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

class Img2Vec():
    def __init__(self, model_name, cuda=False):
        self.device = torch.device("cuda" if cuda else "cpu")
        if model_name.startswith("coca_"):
            self.model, _, self.transform = create_model_and_transforms(model_name, pretrained='mscoco_finetuned_laion2b_s13b_b90k')
        else:
            try:
                self.model = getattr(models, model_name)(pretrained=True).to(self.device)
            except AttributeError:
                try:
                    self.model = timm.create_model(model_name, pretrained=True).to(self.device)
                except RuntimeError as e:
                    if "No pretrained weights exist" in str(e):
                        self.model = timm.create_model(model_name, pretrained=False).to(self.device)
                    else:
                        raise e
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.model.eval()

        # Register a hook to get the output of the penultimate layer
        if 'efficientnet' in model_name:
            self.feature = None
            self.model.classifier[0].register_forward_hook(self._hook_fn)

        if 'convnext' in model_name:
            self.feature = None
            # Register hook at the LayerNorm layer (penultimate layer before the final fc layer)
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.feature = output

    def get_vec(self, img):
        image = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(image)
        embedding_np = self.feature.cpu().numpy().flatten()
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

def pairwise_similarity_distribution(embeddings, labels):
    same_class_sims = []
    diff_class_sims = []

    num_embeddings = len(embeddings)
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if labels[i] == labels[j]:
                same_class_sims.append(sim)
            else:
                diff_class_sims.append(sim)
    
    return same_class_sims, diff_class_sims

def normalize_similarity_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores

def plot_similarity_distribution(same_class_sims, diff_class_sims, model_name, x_range):
    plt.hist(same_class_sims, bins=50, alpha=0.5, label='Same Class', density=True, range=x_range)
    plt.hist(diff_class_sims, bins=50, alpha=0.5, label='Different Class', density=True, range=x_range)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title(f'Similarity Distribution for {model_name}')
    plt.legend()
    plt.show()

def classify_images(model_dict, folder_path, cuda=False):
    images, labels = load_images_from_folder(folder_path)
    results = {}

    for model_display_name, model_name in model_dict.items():
        print(f"Evaluating model {model_name}")
        img2vec = Img2Vec(model_name, cuda=cuda)
        if img2vec.model is None:
            continue

        start_time = time.time()
        embeddings = [img2vec.get_vec(img) for img in images]
        embeddings = [e for e in embeddings if e is not None]
        embeddings = np.array(embeddings)
        processing_time = (time.time() - start_time) / len(images)

        same_class_sims, diff_class_sims = pairwise_similarity_distribution(embeddings, labels)
        same_class_sims = normalize_similarity_scores(same_class_sims)
        diff_class_sims = normalize_similarity_scores(diff_class_sims)

        results[model_display_name] = {
            'same_class_sims': same_class_sims,
            'diff_class_sims': diff_class_sims,
            'processing_time': processing_time
        }

    global_xlim = (0, 1)
    for model_name, result in results.items():
        plot_similarity_distribution(result['same_class_sims'], result['diff_class_sims'], model_name, global_xlim)

    return results

if __name__ == "__main__":
    folder_path = './data-gatter/train_L'

    model_dict = {
        # 'EfficientNetV2': 'efficientnet_v2_s'
        'ConvNeXt': 'convnext_base'
    }

    cuda = torch.cuda.is_available()

    results = classify_images(model_dict, folder_path, cuda)
    for model_name, result in results.items():
        print(f"\nModel: {model_name}")
        print(f"Processing Time per Image: {result['processing_time']}")
