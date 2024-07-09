import os
import ssl
import time
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pinecone import Pinecone, ServerlessSpec
from open_clip import create_model_and_transforms
from torchvision import transforms, models
import timm

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

def classify_images(model_dict, folder_path, cuda=False):
    images, labels = load_images_from_folder(folder_path)
    label_set = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(label_set)}

    results = {}
    for model_display_name, model_name in model_dict.items():
        print(f"Evaluating model {model_name}")
        img2vec = Img2Vec(model_name, cuda=cuda)

        start_time = time.time()
        embeddings = [img2vec.get_vec(img) for img in images]
        embeddings = np.array(embeddings)
        processing_time = (time.time() - start_time) / len(images)

        predicted_labels = []
        for i, embedding in enumerate(embeddings):
            similarities = []
            for j in range(len(images)):
                if i != j:
                    sim = cosine_similarity([embedding], [embeddings[j]])[0][0]
                    similarities.append((sim, labels[j]))

            similarities.sort(reverse=True, key=lambda x: x[0])
            most_similar_label = similarities[0][1]
            predicted_labels.append(most_similar_label)

        true_indices = [label_to_index[label] for label in labels]
        predicted_indices = [label_to_index[label] for label in predicted_labels]

        cm = confusion_matrix(true_indices, predicted_indices)
        report = classification_report(true_indices, predicted_indices, target_names=label_set)

        results[model_display_name] = {
            'confusion_matrix': cm,
            'classification_report': report,
            'processing_time': processing_time
        }

    return results

def visualize_confusion_matrix(cm, model_name, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

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
        print(f"Confusion Matrix:\n{result['confusion_matrix']}")
        print(f"Classification Report:\n{result['classification_report']}")
        print(f"Processing Time per Image: {result['processing_time']}")
        visualize_confusion_matrix(result['confusion_matrix'], model_name, sorted(set(load_images_from_folder(folder_path)[1])))
