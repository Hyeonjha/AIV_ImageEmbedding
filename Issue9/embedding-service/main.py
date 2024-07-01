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

    print("Embedding service started and waiting for messages...", flush=True)

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}", flush=True)
            continue

        image_id = msg.value().decode('utf-8')
        image_path = os.path.join(Config.IMAGE_FOLDER, f"{image_id}.jpg")  # Assuming image extension
        print(f"Processing image ID: {image_id} from path: {image_path}", flush=True)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}", flush=True)
            continue

        try:
            img = Image.open(image_path).convert('RGB')
            embedding = img2vec.get_vec(img)

            data_object = {
                "image_id": image_id,
                "embedding": embedding.tolist()
            }

            client.data_object.create(data_object, "ImageEmbedding", vector=embedding)
            print(f"Successfully stored embedding for image ID: {image_id}", flush=True)

        except Exception as e:
            print(f"Error processing image ID {image_id}: {e}", flush=True)

if __name__ == '__main__':
    start_service()
