from confluent_kafka import Consumer
import weaviate
import os
from PIL import Image
from Img2Vec import Img2Vec
from config import Config

def start_service():
    client = weaviate.Client(Config.WEAVIATE_URL)
    img2vec = Img2Vec()

    consumer = Consumer({
        'bootstrap.servers': Config.KAFKA_BROKER,
        'group.id': 'embedding-service',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe(['image-uploads'])

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        image_id = msg.value().decode('utf-8')
        image_path = os.path.join(Config.IMAGE_FOLDER, f"{image_id}.jpg")  # Assuming image extension
        img = Image.open(image_path).convert('RGB')
        embedding = img2vec.get_vec(img)

        data_object = {
            "image_id": image_id,
            "embedding": embedding.tolist()
        }

        client.data_object.create(data_object, "ImageEmbedding", vector=embedding)

if __name__ == '__main__':
    start_service()
