import os

class Config:
    KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')
    WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://weaviate:8080')
    IMAGE_FOLDER = '/mnt/data/images'
