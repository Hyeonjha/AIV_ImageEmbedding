### Programmer : Hyeon Ji Ha
### Date : Jun 14 2024
### Purpose : SQLAlchemy 모델을 정의
###           

from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class ImageEmbedding(Base):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    embedding_vector = Column(Vector(1000))  # pgvector 사용 - 모델별 조정
