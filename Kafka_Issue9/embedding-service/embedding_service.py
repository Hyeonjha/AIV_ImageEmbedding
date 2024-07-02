import os
import weaviate
import torch
from torchvision import models, transforms
from kafka import KafkaConsumer
from sqlalchemy import create_engine, Column, String, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from PIL import Image as PILImage

# 데이터베이스 URL 환경 변수
DATABASE_URL = os.getenv('DATABASE_URI')

# SQLAlchemy 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 이미지 테이블 정의
class Image(Base):
    __tablename__ = "images"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    size = Column(Integer)
    path = Column(String)
    hash = Column(String)
    image_metadata = Column(JSON)  # 변경된 부분

# 테이블 생성
Base.metadata.create_all(bind=engine)

# Kafka 소비자 설정
consumer = KafkaConsumer('image_topic', bootstrap_servers=os.getenv('KAFKA_BROKER'))

# Weaviate 클라이언트 설정
client = weaviate.Client(os.getenv('WEAVIATE_URL'))

# ConvNeXT 모델 설정
model = models.convnext_base(pretrained=True)
model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 메시지 소비 및 임베딩 생성
for message in consumer:
    image_id = message.value.decode('utf-8')  # Kafka 메시지에서 이미지 ID 추출
    db = SessionLocal()  # 데이터베이스 연결

    # 이미지 정보 조회
    db_image = db.query(Image).filter(Image.id == image_id).first()
    db.close()

    if db_image:
        img_path = db_image.path  # 이미지 파일 경로
        image = PILImage.open(img_path)  # 이미지 열기

        # 이미지 전처리
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # 모델을 사용하여 임베딩 생성
        with torch.no_grad():
            embedding = model(input_batch).numpy().flatten().tolist()

        # Weaviate에 임베딩 저장
        client.data_object.create(
            data_object={
                "uuid": image_id,
                "embedding": embedding,
            },
            class_name="ImageEmbedding"
        )
