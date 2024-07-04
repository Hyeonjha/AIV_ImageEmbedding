import os
import hashlib
import uuid
import io
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from kafka import KafkaProducer
from pydantic import BaseModel
import weaviate
from typing import List
from PIL import Image as PILImage
import torch
from torchvision import models, transforms

# 데이터베이스 URL 환경 변수
DATABASE_URL = os.getenv('DATABASE_URI')
WEAVIATE_URL = os.getenv('WEAVIATE_URL')

# SQLAlchemy 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FastAPI 앱 생성
app = FastAPI()

# 이미지 테이블 정의
class Image(Base):
    __tablename__ = "images"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    size = Column(Integer)
    path = Column(String)
    hash = Column(String)
    image_metadata = Column(JSON)

# 테이블 생성
Base.metadata.create_all(bind=engine)

# Kafka 프로듀서 설정
producer = KafkaProducer(bootstrap_servers=os.getenv('KAFKA_BROKER'))

# 응답 모델 정의
class UploadResponse(BaseModel):
    message: str
    image_id: str

class SimilarImageResponse(BaseModel):
    image_id: str
    score: float

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이미지 업로드 엔드포인트
@app.post("/upload/", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()  # 이미지 파일을 Byte로 읽음
    hash = hashlib.sha256(contents).hexdigest()  # 이미지 해시 생성
    size = len(contents)  # 파일 크기 계산
    extension = file.filename.split('.')[-1]  # 파일 확장자 추출
    filename = f"{hash}-{size}.{extension}"  # 파일명 생성
    filepath = f"/mnt/data/images/{filename}"  # 파일 저장 경로 설정
    
    # 파일 저장
    with open(filepath, "wb") as f:
        f.write(contents)

    image_id = str(uuid.uuid4())  # 이미지 ID 생성 (UUID)
    db = SessionLocal()  # 데이터베이스 연결
    image_metadata = {"original_filename": file.filename}  # 메타데이터 설정
    
    # 이미지 정보 데이터베이스에 저장
    db_image = Image(id=image_id, name=file.filename, size=size, path=filepath, hash=hash, image_metadata=image_metadata)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    db.close()

    # Kafka에 메시지 전송
    producer.send('image_topic', value=image_id.encode('utf-8'))

    # 응답 변환
    return JSONResponse(content={"message": "Image uploaded successfully", "image_id": image_id})

# 모델 설정
model = models.convnext_base(pretrained=True)
model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 유사 이미지 검색 엔드포인트
@app.post("/search_similar/", response_model=List[SimilarImageResponse])
async def search_similar(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = PILImage.open(io.BytesIO(contents))

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = model(input_batch).numpy().flatten().tolist()

        if not WEAVIATE_URL:
            raise ValueError("WEAVIATE_URL environment variable is not set")

        client = weaviate.Client(WEAVIATE_URL)

        query_result = client.query.get("ImageEmbedding", ["uuid", "_additional { score }"]) \
            .with_near_vector({"vector": embedding}) \
            .with_limit(10) \
            .do()

        results = []
        for result in query_result["data"]["Get"]["ImageEmbedding"]:
            results.append(SimilarImageResponse(image_id=result["uuid"], score=result["_additional"]["score"]))

        return results

    except Exception as e:
        print(f"Error during searching similar images: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})