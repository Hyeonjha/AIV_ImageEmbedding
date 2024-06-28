import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://user:password@db:5432/imagedb')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')
