import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import ssl
import matplotlib.pyplot as plt
import seaborn as sns
from open_clip import create_model_and_transforms
import timm
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

def load_images_from_folder(folder, label_map=None):
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
                if label_map:
                    labels.append(label_map[class_folder_name])
                else:
                    labels.append(class_folder_name)
    return images, labels

def load_test_images(folder):
    images = []
    filenames = []
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.splitext(filename)[1].lower() in valid_image_extensions:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            filenames.append(filename)
    return images, filenames

def classify_images(model_dict, train_folder, test_folder, cuda=False):
    train_images, train_labels = load_images_from_folder(train_folder)
    print(f"Loaded {len(train_images)} training images from {train_folder}")
    
    test_images, test_filenames = load_test_images(test_folder)
    print(f"Loaded {len(test_images)} testing images from {test_folder}")

    label_set = sorted(set(train_labels))
    label_to_index = {label: idx for idx, label in enumerate(label_set)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    results = {}
    for model_display_name, model_name in model_dict.items():
        print(f"Evaluating model {model_name}")
        if model_name.startswith("coca_"):
            img2vec = CoCaImg2Vec(model_name, pretrained='mscoco_finetuned_laion2b_s13b_b90k', cuda=cuda)
        else:
            img2vec = Img2Vec(model_name, cuda=cuda)
 
        #############
        print(f"Extracting features for training images using {model_display_name}...")
        train_embeddings = []
        for i, img in enumerate(train_images):
            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(train_images)} training images")
            train_embeddings.append(img2vec.get_vec(img))
        train_embeddings = np.array(train_embeddings)
        ##############

        start_time = time.time()
        # train_embeddings = [img2vec.get_vec(img) for img in train_images]
        # train_embeddings = np.array(train_embeddings)
        processing_time = (time.time() - start_time) / len(train_images)

        ###############
        print(f"Extracting features for testing images using {model_display_name}...")
        test_embeddings = []
        for i, img in enumerate(test_images):
            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(test_images)} testing images")
            test_embeddings.append(img2vec.get_vec(img))
        ##########

        # test_embeddings = [img2vec.get_vec(img) for img in test_images]

        print(f"Classifying for testing images using {model_display_name}...")
        predicted_labels = []
        for test_embedding in test_embeddings:
            similarities = [cosine_similarity([test_embedding], [train_embedding])[0][0] for train_embedding in train_embeddings]
            most_similar_index = np.argmax(similarities)
            predicted_labels.append(train_labels[most_similar_index])

            if i % 100 == 0 and i > 0:  #################
                print(f"Classified {i}/{len(test_images)} testing images")


        true_labels = [filename.split('_')[0] for filename in test_filenames]  # Assuming the true label is encoded in the filename

        true_indices = [label_to_index[label] for label in true_labels]
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

if __name__ == "__main__":                                                                         # TRAIN_SET_HL_NR_NoSize
    train_folder = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/TRAIN_SET_A'  # TRAIN_SET_A : 전체 데이터  TRAIN_SET_R : RING 기준  TRAIN_SET_NR : DOT(42) 기준
    test_folder = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/TEST_SET_A'
    model_dict = {
        'ConvNeXt': 'convnext_base',
        'ResNet18': 'resnet18',
        'ResNet50': 'resnet50'
    }

    cuda = torch.cuda.is_available()

    results = classify_images(model_dict, train_folder, test_folder, cuda)
    for model_name, result in results.items():
        print(f"\nModel: {model_name}")
        print(f"Confusion Matrix:\n{result['confusion_matrix']}")  
        print(f"Classification Report:\n{result['classification_report']}")
        print(f"Processing Time per Image: {result['processing_time']}")
        visualize_confusion_matrix(result['confusion_matrix'], model_name, sorted(set(load_images_from_folder(train_folder)[1])))
