import time
import numpy as np
import os
from PIL import Image
import weaviate
from Img2Vec import Img2Vec

# Weaviate 클라이언트 설정
client = weaviate.Client(url="http://localhost:8080")

# 스키마 정의
class_obj = {
    "class": "ImageEmbedding",
    "vectorizer": "none",
    "properties": [
        {
            "name": "image_path",
            "dataType": ["string"]
        },
        {
            "name": "label",
            "dataType": ["string"]
        }
    ]
}

# 클래스 존재 여부 확인 및 생성
def ensure_class_exists(client, class_obj):
    try:
        existing_classes = client.schema.get()['classes']
        existing_class_names = [cls['class'] for cls in existing_classes]
        if class_obj['class'] not in existing_class_names:
            client.schema.create_class(class_obj)
        else:
            print(f"Class {class_obj['class']} already exists.")
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        print(f"Error checking/creating class: {e}")

ensure_class_exists(client, class_obj)

img2vec = Img2Vec()

# 데이터 삽입 성능 테스트
insert_times = []
folder_path = 'data-gatter/train_L'
image_paths = []

# 폴더 내 모든 이미지 경로 수집
for label in os.listdir(folder_path):
    label_path = os.path.join(folder_path, label)
    if not os.path.isdir(label_path):
        continue
    for filename in os.listdir(label_path):
        img_path = os.path.join(label_path, filename)
        if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_paths.append((img_path, label))

# 이미지 삽입 성능 테스트 반복
for i in range(min(20, len(image_paths))):  # 10000
    img_path, label = image_paths[i]
    img = Image.open(img_path).convert('RGB')
    embedding = img2vec.get_vec(img)
    data_object = {
        "image_path": img_path,
        "label": label,
    }
    start_time = time.time()
    client.data_object.create(data_object, "ImageEmbedding", vector=embedding)
    end_time = time.time()
    insert_times.append(end_time - start_time)

insert_mean = np.mean(insert_times)
insert_std = np.std(insert_times)

print(f"Insert - Mean time: {insert_mean}, Std time: {insert_std}")

# 검색 성능 테스트
search_times = []
test_image_path = 'data-gatter/test'
test_image_paths = [os.path.join(test_image_path, f) for f in os.listdir(test_image_path) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]

# 검색 성능 테스트 반복
for i in range(min(20, len(test_image_paths))):  # 10000
    img_path = test_image_paths[i]
    img = Image.open(img_path).convert('RGB')
    embedding = img2vec.get_vec(img)
    start_time = time.time()
    query_result = client.query.get("ImageEmbedding", ["image_path", "label"])\
        .with_near_vector({"vector": embedding, "certainty": 0.8})\
        .with_limit(5)\
        .do()
    end_time = time.time()
    search_times.append(end_time - start_time)

search_mean = np.mean(search_times)
search_std = np.std(search_times)

print(f"Search - Mean time: {search_mean}, Std time: {search_std}")
