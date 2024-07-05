import os
import hashlib
import uuid
import io
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
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
from dotenv import load_dotenv
from fastapi.responses import FileResponse

# 환경 변수 로드
load_dotenv()

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

# Weaviate 스키마 설정
def create_weaviate_schema():
    client = weaviate.Client(WEAVIATE_URL)

    # 현재 스키마 확인
    current_schema = client.schema.get()
    print("Current schema:", current_schema)

    schema = {
        "classes": [
            {
                "class": "ImageEmbedding",
                "description": "A class to store image embeddings",
                "vectorizer": "none",
           #     "vectorIndexType": "hnsw",  # HNSW 인덱스 사용
                "properties": [
                    {
                        "name": "uuid",
                        "dataType": ["string"],
                        "description": "The UUID of the image"
                    },
                    {
                        "name": "embedding",
                        "dataType": ["number[]"],
                        "description": "The embedding vector of the image"
                    }
                ]
            }
        ]
    }
    """
    client.schema.delete_class("ImageEmbedding")  # 기존 클래스 삭제 (있을 경우)
    client.schema.create(schema)
    print("Weaviate schema created")
"""
    # 기존 클래스가 있다면 삭제
    if "ImageEmbedding" in [c['class'] for c in current_schema['classes']]:
        client.schema.delete_class("ImageEmbedding")
    
    # 새 스키마 생성
    client.schema.create(schema)
    print("Weaviate schema created")

    # 생성된 스키마 확인
    new_schema = client.schema.get()
    print("New schema:", new_schema)

# FastAPI lifespan 이벤트 핸들러로 스키마 생성
@app.on_event("startup")
async def startup_event():
    create_weaviate_schema()

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

def weaviate_certainty_to_cosine(certainty):
    return 2 * certainty - 1

# 유사 이미지 검색 엔드포인트
@app.post("/search_similar/", response_model=List[SimilarImageResponse])
async def search_similar(file: UploadFile = File(...), threshold: float = 0.8, top_n: int = 5):
    try:
        contents = await file.read()
        image = PILImage.open(io.BytesIO(contents))

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = model(input_batch).numpy().flatten().tolist()

        #print("Embedding:", embedding)  # 디버그를 위한 임베딩 출력

        print(f"Search embedding: {embedding[:10]}...")  # 처음 10개 요소만 출력

        if not WEAVIATE_URL:
            raise ValueError("WEAVIATE_URL environment variable is not set")

        client = weaviate.Client(WEAVIATE_URL)
        
        
        we_result = client.query.get("ImageEmbedding", ["uuid"]).do()
        print(f"we_result : {we_result}")

        we_result2 = client.query.get("ImageEmbedding", ["uuid", "_additional { distance }"]) \
            .with_near_vector({"vector": embedding}) \
            .with_limit(top_n) \
            .do()
        print(f"we_result2 : {we_result2}")

        query_result = client.query.get("ImageEmbedding", ["uuid", "_additional { certainty }"]) \
            .with_near_vector({"vector": embedding}) \
            .with_limit(top_n) \
            .do()
        
        """

        # 간단한 쿼리로 테스트
        test_query = client.query.get("ImageEmbedding", ["uuid"]).do()
        print("Test query result:", test_query)

        # 벡터 검색 쿼리
        query_result = client.query.get("ImageEmbedding", ["uuid"]) \
            .with_near_vector({"vector": embedding}) \
            .with_limit(top_n) \
            .with_additional(["distance"]) \
            .do()
        """


        print("Query result:", query_result)  # 디버그를 위한 로그 추가

        if 'errors' in query_result:
            raise HTTPException(status_code=500, detail=query_result['errors'])

       
        results = []
        for item in query_result['data']['Get']['ImageEmbedding']:
            print("start searching similarity")

            certainty = item['_additional']['certainty']
            cosine_similarity = weaviate_certainty_to_cosine(certainty)

            print(f"sim : {cosine_similarity}")

            if cosine_similarity >= threshold:
                results.append(SimilarImageResponse(image_id=item['uuid'], score=cosine_similarity))
        """
        results = []
        for item in query_result['data']['Get']['ImageEmbedding']:
            distance = item['_additional']['distance']
            similarity = 1 - distance  # 거리를 유사도로 변환
            if similarity >= threshold:
                results.append(SimilarImageResponse(image_id=item['uuid'], score=similarity))
        """
        print("Results:", results)  # 디버그를 위한 로그 추가

        return results

    except Exception as e:
        print(f"Error during searching similar images: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
    
# 이미지 파일 제공 엔드포인트
@app.get("/images/{image_id}")
async def get_image(image_id: str):
    db = SessionLocal()
    db_image = db.query(Image).filter(Image.id == image_id).first()
    db.close()
    if db_image:
        return FileResponse(db_image.path)
    return JSONResponse(status_code=404, content={"message": "Image not found"})