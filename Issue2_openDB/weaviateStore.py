# weaviate_store.py
import weaviate
import os
from PIL import Image
from Img2Vec import Img2Vec

# Weaviate 클라이언트 설정
client = weaviate.Client(
    url="http://localhost:8080"
)

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

def save_image_embedding(folder_path):
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
                data_object = {
                    "image_path": img_path,
                    "label": label,
                }
                client.data_object.create(data_object, "ImageEmbedding", vector=embedding)

if __name__ == "__main__":
    folder_path = 'data-gatter/train_L'  # 이미지 폴더 경로
    save_image_embedding(folder_path)
