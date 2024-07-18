import os
import shutil
import torch
import ssl
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np

# SSL 인증서 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# 이미지 폴더 경로 설정
base_dir = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/TRAIN_SET_A'
output_dir = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/TRAIN_SET_KR'  # TRAIN_SET_NR  TRAIN_SET_R

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

def cluster_and_select_representatives(image_features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(image_features)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    representatives = []
    for i in range(num_clusters):
        cluster_members = np.where(labels == i)[0]
        cluster_features = image_features[cluster_members]
        center = cluster_centers[i]
        closest_member = cluster_members[np.argmin(np.linalg.norm(cluster_features - center, axis=1))]
        representatives.append(closest_member)

    return representatives

def classify_images(class_path, min_count):
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and os.path.splitext(f)[1].lower() in valid_image_extensions]

    image_features = []
    # for img in images:
    #     img_path = os.path.join(class_path, img)
    #     features = extract_features(img_path)
    #     with Image.open(img_path) as image:
    #         width, height = image.size
    #         aspect_ratio = width / height
    #         size_features = [width, height, aspect_ratio]
    #     combined_features = np.concatenate((features, size_features))
    #     image_features.append(combined_features)

    for img in images:
        img_path = os.path.join(class_path, img)
        features = extract_features(img_path)
        image_features.append(features)

    image_features = np.array(image_features)

    # K-means 클러스터링을 사용하여 대표 이미지 선택
    representative_indices = cluster_and_select_representatives(image_features, min_count-1)  #####
    unique_images = [images[i] for i in representative_indices]

    # 유사도 행렬 계산
    similarity_matrix = calculate_similarity(image_features)

    # 유사도가 가장 낮은 이미지 추가 선택
    unique_indices = set(representative_indices)
    while len(unique_indices) < min_count * 2 - 1:  ########
        remaining_indices = [i for i in range(len(images)) if i not in unique_indices]
        if remaining_indices:
            remaining_similarities = similarity_matrix[remaining_indices, :][:, list(unique_indices)].mean(axis=1)
            next_index = remaining_indices[np.argmin(remaining_similarities)]
            unique_indices.add(next_index)
        else:
            break

    additional_unique_images = [images[i] for i in unique_indices if i not in representative_indices]
    unique_images.extend(additional_unique_images[:min_count])

    remaining_indices = [i for i in range(len(images)) if i not in unique_indices]
    similar_images = [images[i] for i in remaining_indices]

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
