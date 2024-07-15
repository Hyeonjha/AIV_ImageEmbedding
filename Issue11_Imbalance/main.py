import os
import shutil
import ssl
from PIL import Image
from torchvision import transforms, models
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# SSL 인증서 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 임베딩 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)  # convnext_base  resnet18
model = model.to(device)
model.eval()

# 이미지 폴더 경로 설정
base_dir = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/downloaded'
output_dir = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/Issue11_Imbalance/classified_'

# 클래스별 이미지 경로 수집
class_dirs = {d: os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))}

# 각 클래스의 이미지 리스트 수집
class_images = {cls: [os.path.join(path, img) for img in os.listdir(path) if img.endswith(('jpg', 'jpeg', 'png'))] for cls, path in class_dirs.items()}

# 가장 적은 이미지 수를 가진 클래스의 이미지 개수 찾기
min_count = min(len(images) for images in class_images.values())

print(f"min count : {min_count}")

# 임베딩 추출 함수
def extract_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.cpu().numpy().flatten()

# 각 클래스별로 가장 unique 한 이미지 선택 및 유사한 이미지는 similar_image 폴더로 이동
for cls, images in class_images.items():
    print(f"cls : {cls}")
    embeddings = [extract_embedding(img) for img in images]
    unique_indices = []
    similar_indices = []
    
    # 코사인 유사도를 계산하여 유사 이미지 그룹화
    similarity_matrix = cosine_similarity(embeddings)
    for i, row in enumerate(similarity_matrix):
        if i not in similar_indices:
            unique_indices.append(i)
            for j, sim in enumerate(row):
                if sim > 0.8 and j != i:  # 유사도 임계값을 0.8로 설정
                    similar_indices.append(j)


    # unique 이미지와 similar 이미지를 분리
    unique_indices = unique_indices[:min_count]
    similar_indices = set(similar_indices) - set(unique_indices)
    
    # 최소 unique 이미지 개수 보장
    if len(unique_indices) < min_count:
        remaining = set(range(len(images))) - set(unique_indices) - similar_indices
        unique_indices.extend(list(remaining)[:min_count - len(unique_indices)])
    
    # unique 이미지를 새로운 클래스 폴더에, 유사 이미지를 새로운 similar_image 폴더에 저장
    unique_class_dir = os.path.join(output_dir, cls)
    similar_class_dir = os.path.join(unique_class_dir, 'similar_image')
    os.makedirs(unique_class_dir, exist_ok=True)
    os.makedirs(similar_class_dir, exist_ok=True)
    
    for idx in unique_indices:
        shutil.copy(images[idx], os.path.join(unique_class_dir, os.path.basename(images[idx])))
    
    for idx in similar_indices:
        shutil.copy(images[idx], os.path.join(similar_class_dir, os.path.basename(images[idx])))

print("이미지 분류가 완료되었습니다.")