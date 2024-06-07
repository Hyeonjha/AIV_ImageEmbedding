import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import certifi
import os
import ssl
import time

# SSL 인증서 설정
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

class Img2Vec():
    RESNET_OUTPUT_SIZES = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }

    EFFICIENTNET_OUTPUT_SIZES = {
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2304,
        'efficientnet_b7': 2560
    }

    def __init__(self, cuda=False, model='resnet18', layer='default', layer_output_size=512, gpu=0):
        self.device = torch.device(f"cuda:{gpu}" if cuda and torch.cuda.is_available() else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        if type(img) == list:
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(len(img), self.layer_output_size).to(self.device)
            elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7).to(self.device)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1).to(self.device)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg']:
                    return my_embedding.cpu().numpy()[:, :]
                elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
                    return torch.mean(my_embedding, (2, 3), True).cpu().numpy()[:, :, 0, 0]
                else:
                    return my_embedding.cpu().numpy()[:, :, 0, 0]
        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(1, self.layer_output_size).to(self.device)
            elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
                my_embedding = torch.zeros(1, self.layer_output_size, 7, 7).to(self.device)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1).to(self.device)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg']:
                    return my_embedding.cpu().numpy()[0, :]
                elif self.model_name == 'densenet':
                    return torch.mean(my_embedding, (2, 3), True).cpu().numpy()[0, :, 0, 0]
                else:
                    return my_embedding.cpu().numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        if model_name.startswith('resnet') and not model_name.startswith('resnet-'):
            model = getattr(models, model_name)(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)
            return model, layer
        elif model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'vgg':
            model = models.vgg11_bn(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = model.classifier[-1].in_features
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'densenet':
            model = models.densenet121(pretrained=True)
            if layer == 'default':
                layer = model.features[-1]
                self.layer_output_size = model.classifier.in_features
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer

        elif "efficientnet" in model_name:
            if model_name == "efficientnet_b0":
                model = models.efficientnet_b0(pretrained=True)
            elif model_name == "efficientnet_b1":
                model = models.efficientnet_b1(pretrained=True)
            elif model_name == "efficientnet_b2":
                model = models.efficientnet_b2(pretrained=True)
            elif model_name == "efficientnet_b3":
                model = models.efficientnet_b3(pretrained=True)
            elif model_name == "efficientnet_b4":
                model = models.efficientnet_b4(pretrained=True)
            elif model_name == "efficientnet_b5":
                model = models.efficientnet_b5(pretrained=True)
            elif model_name == "efficientnet_b6":
                model = models.efficientnet_b6(pretrained=True)
            elif model_name == "efficientnet_b7":
                model = models.efficientnet_b7(pretrained=True)
            else:
                raise KeyError('Un support %s.' % model_name)

            if layer == 'default':
                layer = model.features
                self.layer_output_size = self.EFFICIENTNET_OUTPUT_SIZES[model_name]
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)

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

def knn_classify(embeddings, labels, query_embedding, k):
    # embeddings와 query_embedding 간의 유사도 계산
    similarities = [cosine_similarity([query_embedding], [embedding])[0][0] for embedding in embeddings]
    
    # 유사도와 레이블을 함께 저장
    similarity_label_pairs = list(zip(similarities, labels))
    
    # 유사도 기준으로 내림차순 정렬
    similarity_label_pairs.sort(reverse=True, key=lambda x: x[0])
    
    # k개의 가장 유사한 이웃 선택
    nearest_neighbors = similarity_label_pairs[:k]
    
    # 가장 많이 등장한 레이블 선택
    neighbor_labels = [label for _, label in nearest_neighbors]
    most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
    
    return most_common_label

def classify_images_knn(model_names, folder_path, k_values, cuda=False):
    # 이미지와 레이블 불러오기
    images, labels = load_images_from_folder(folder_path)

    results = {}
    for model_name in model_names:
        print(f"Evaluating model {model_name}")
        # 모델 초기화
        img2vec = Img2Vec(cuda=cuda, model=model_name)

        # 임베딩 추출
        start_time = time.time()
        embeddings = [img2vec.get_vec(img) for img in images]
        embeddings = np.array(embeddings)
        processing_time = (time.time() - start_time) / len(images)

        best_accuracy = 0
        best_k = None
        for k in k_values:
            # KNN 알고리즘을 사용하여 분류
            correct_classifications = 0
            incorrect_classifications = 0

            for i, query_embedding in enumerate(embeddings):
                # Leave-one-out 방식으로 평가
                leave_out_embeddings = np.delete(embeddings, i, axis=0)
                leave_out_labels = np.delete(labels, i, axis=0)
                
                predicted_label = knn_classify(leave_out_embeddings, leave_out_labels, query_embedding, k)
                
                if predicted_label == labels[i]:
                    correct_classifications += 1
                else:
                    incorrect_classifications += 1

            # 정확도 계산
            total_images = correct_classifications + incorrect_classifications
            accuracy = correct_classifications / total_images

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        results[model_name] = {
            'best_k': best_k,
            'best_accuracy': best_accuracy,
            'processing_time': processing_time,
            'correct_classifications': correct_classifications,
            'incorrect_classifications': incorrect_classifications
        }

    return results

if __name__ == "__main__":
    folder_path = './data-gatter/train_L'
    model_names = ['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b7']
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    cuda = torch.cuda.is_available()

    results = classify_images_knn(model_names, folder_path, k_values, cuda)
    for model_name, result in results.items():
        print(f"\nModel: {model_name}")
        print(f"Best K: {result['best_k']}")
        print(f"Accuracy: {result['best_accuracy']}")
        print(f"Processing Time per Image: {result['processing_time']} seconds")
        print(f"Correct Classifications: {result['correct_classifications']}")
        print(f"Incorrect Classifications: {result['incorrect_classifications']}")
