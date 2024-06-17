import weaviate
from PIL import Image
from Img2Vec import Img2Vec

# Weaviate 클라이언트 설정
client = weaviate.Client(
    url="http://localhost:8080"
)

def find_similar_images(input_image_path, threshold=0.8, top_n=5):
    img = Image.open(input_image_path).convert('RGB')
    img2vec = Img2Vec()
    input_embedding = img2vec.get_vec(img)

    query_result = client.query.get("ImageEmbedding", ["image_path", "label"])\
        .with_near_vector({"vector": input_embedding, "certainty": threshold})\
        .with_limit(top_n)\
        .do()

    return [(item['image_path'], item['label']) for item in query_result['data']['Get']['ImageEmbedding']]

if __name__ == "__main__":
    input_image_path = 'data-gatter/testcopy/dot_381751.jpg'  # 입력 이미지 경로
    similar_images = find_similar_images(input_image_path)
    print(similar_images)
