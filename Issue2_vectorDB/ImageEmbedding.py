from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ImageEmbedding(Base):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True)
    image_path = Column(String, nullable=False)
    label = Column(String, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    embedding_vector = Column(ARRAY(Float), nullable=False)

class HNSWIndex(Base):
    __tablename__ = 'hnsw_index'
    id = Column(Integer, primary_key=True)
    index_data = Column(LargeBinary, nullable=False)

# PostgreSQL 설정
DATABASE_URL = "postgresql://postgres:aiv11011@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
