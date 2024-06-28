import weaviate
from PIL import Image
from Img2Vec import Img2Vec
import numpy as np

# Weaviate 클라이언트 설정
client = weaviate.Client(
    url="http://localhost:8080"
)

def weaviate_certainty_to_cosine(certainty):
    return 2 * certainty - 1

def find_similar_images(input_image_path, threshold=0.8, top_n=5):
    img = Image.open(input_image_path).convert('RGB')
    img2vec = Img2Vec()
    input_embedding = img2vec.get_vec(img)

    query_result = client.query.get("ImageEmbedding", ["image_path", "label", "_additional {certainty}"])\
        .with_near_vector({"vector": input_embedding})\
        .with_limit(top_n)\
        .do()

    similar_images = []
    for item in query_result['data']['Get']['ImageEmbedding']:
        certainty = item['_additional']['certainty']
        cosine_similarity = weaviate_certainty_to_cosine(certainty)
        if cosine_similarity >= threshold:
            similar_images.append((item['image_path'], item['label'], cosine_similarity))

    return similar_images


if __name__ == "__main__":
    input_image_path = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/test/bubble_380033.jpg'  # 입력 이미지 경로
    similar_images = find_similar_images(input_image_path)
    for image_path, label, similarity in similar_images:
        print(f"Image Path: {image_path}, Label: {label}, Similarity: {similarity:.4f}")
