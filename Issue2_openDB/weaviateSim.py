import numpy as np
from PIL import Image
import weaviate
from Img2Vec import Img2Vec
import os

# Weaviate 클라이언트 설정
client = weaviate.Client(url="http://localhost:8080")

# Img2Vec 초기화
img2vec = Img2Vec()

# 코사인 유사도 함수
def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

# Weaviate certainty 값을 코사인 유사도로 변환하는 함수
def weaviate_certainty_to_cosine(certainty):
    return 2 * certainty - 1

# 테스트 이미지 경로
test_image_path = 'data-gatter/test'
test_image_files = [os.path.join(test_image_path, f) for f in os.listdir(test_image_path) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]

# 성능 측정
similarities_weaviate = []  # weaviate에서 제공한 유사도 값을 저장하는 리스트
similarities_manual = []  # 직접 계산한 코사인 유사도 값 저장하는 리스트

for img_path in test_image_files:
    img = Image.open(img_path).convert('RGB')
    input_embedding = img2vec.get_vec(img)

    # Weaviate에서 유사한 이미지 검색
    query_result = client.query.get("ImageEmbedding", ["image_path", "label", "_additional {certainty}"])\
        .with_near_vector({"vector": input_embedding})\
        .with_limit(5)\
        .do()

    for result in query_result['data']['Get']['ImageEmbedding']:
        weaviate_certainty = result['_additional']['certainty']
        weaviate_similarity = weaviate_certainty_to_cosine(weaviate_certainty)  # certainty를 이용하여 코사인 유사도로 변환
        result_image_path = result['image_path']
        result_img = Image.open(result_image_path).convert('RGB')
        result_embedding = img2vec.get_vec(result_img)

        manual_similarity = cosine_similarity_manual(input_embedding, result_embedding)

        similarities_weaviate.append(weaviate_similarity)
        similarities_manual.append(manual_similarity)

# 결과 비교
similarities_weaviate = np.array(similarities_weaviate)
similarities_manual = np.array(similarities_manual)

mean_diff = np.mean(np.abs(similarities_weaviate - similarities_manual))
std_diff = np.std(np.abs(similarities_weaviate - similarities_manual))

print(f"Mean difference between Weaviate and manual cosine similarity: {mean_diff:.4f}")
print(f"Standard deviation of difference: {std_diff:.4f}")