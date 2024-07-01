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
    folder_path = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L'  # 이미지 폴더 경로
    save_image_embedding(folder_path)

"""
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378028.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378029.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378030.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378044.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378064.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378065.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378066.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378067.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378081.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378082.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378092.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378097.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378098.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378144.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/378408.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/BUBBLE/379197.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/DAMAGE/378274.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/DAMAGE/378019.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/DAMAGE/378231.jpg' http://localhost:5001/upload
curl -F 'file=@/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L/DAMAGE/378567.jpg' http://localhost:5001/upload





app.py
from flask import Flask, request, jsonify
import os
import hashlib
from models import db, Image
from config import Config
from confluent_kafka import Producer
import uuid

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

producer = Producer({'bootstrap.servers': app.config['KAFKA_BROKER']})

UPLOAD_FOLDER = '/mnt/data/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            filename = file.filename
            file_content = file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            file_size = len(file_content)
            ext = filename.rsplit('.', 1)[1].lower()
            saved_filename = f"{file_hash}-{file_size}.{ext}"
            file_path = os.path.join(UPLOAD_FOLDER, saved_filename)
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            image = Image(id=str(uuid.uuid4()), hash=file_hash, filename=saved_filename, size=file_size, path=file_path)
            db.session.add(image)
            db.session.commit()

            producer.produce('image-uploads', key=image.id, value=image.id)
            producer.flush()

            return jsonify({'id': image.id}), 201
    except Exception as e:
        print(f"Error: {e}", flush=True)
        db.session.rollback()  # 오류가 발생하면 트랜잭션 롤백
        return 'Internal Server Error', 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

"""