# classify unique data

import os
import shutil
import torch
import ssl
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# SSL 인증서 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# 이미지 폴더 경로 설정
# '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/downloaded' 
base_dir =  '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L'
output_dir = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/classified_train_L_2'

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ResNet18 모델 로드
model = models.resnet18(pretrained=True)
model.eval()

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model(image).squeeze().numpy()
    return features

def calculate_similarity(features_list):
    similarity_matrix = cosine_similarity(features_list)
    return similarity_matrix

def classify_images(class_path, min_count):
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and os.path.splitext(f)[1].lower() in valid_image_extensions]
    
    image_features = []
    image_sizes = []
    for img in images:
        img_path = os.path.join(class_path, img)
        features = extract_features(img_path)
        with Image.open(img_path) as image:
            width, height = image.size
        size_info = (width, height)
        image_features.append((features, size_info))

    # Similarity matrix calculation using both feature and size info
    features_array = np.array([f[0] for f in image_features])
    size_array = np.array([f[1] for f in image_features])
    combined_features = np.hstack((features_array, size_array))

    similarity_matrix = calculate_similarity(combined_features)

    unique_images = []
    similar_images = []

    # 초기 unique_images 설정
    initial_indices = np.argsort(np.sum(similarity_matrix, axis=1))[:min_count]
    unique_images = [images[i] for i in initial_indices]

    remaining_indices = [i for i in range(len(images)) if i not in initial_indices]

    for idx in remaining_indices:
        similarities = [similarity_matrix[idx][i] for i in range(len(images)) if images[i] in unique_images]
        max_similarity = max(similarities) if similarities else 0
        if max_similarity < 0.9:
            if len(unique_images) < min_count:
                unique_images.append(images[idx])
            else:
                similar_images.append(images[idx])
        else:
            similar_images.append(images[idx])

    # 고유 이미지 수를 Min_count로 고정
    if len(unique_images) > min_count:
        similar_images.extend(unique_images[min_count:])
        unique_images = unique_images[:min_count]

    return unique_images, similar_images

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    image_counts = {cls: len(os.listdir(os.path.join(base_dir, cls))) for cls in class_dirs}
    min_count = min(image_counts.values())

    for cls in class_dirs:
        class_path = os.path.join(base_dir, cls)
        output_class_path = os.path.join(output_dir, cls)
        similar_output_path = os.path.join(output_class_path, 'similar_image')

        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
        if not os.path.exists(similar_output_path):
            os.makedirs(similar_output_path)

        unique_images, similar_images = classify_images(class_path, min_count=min_count)

        for img in unique_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_class_path, img)
            shutil.copyfile(src, dst)

        for img in similar_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(similar_output_path, img)
            shutil.copyfile(src, dst)

        print(f"Class {cls}: Unique Images = {len(unique_images)}, Similar Images = {len(similar_images)}")

if __name__ == "__main__":
    main()
