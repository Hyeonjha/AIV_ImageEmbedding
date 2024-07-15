from locust import HttpUser, TaskSet, task, between
from io import BytesIO
from PIL import Image
import random
import numpy as np
import time
import weaviate

WEAVIATE_URL = "http://weaviate:8080"

def weaviate_certainty_to_cosine(certainty):
    return 2 * certainty - 1

def generate_random_vector(dim=1000):
    return np.random.rand(dim).astype(np.float32)
################

def generate_random_image(width, height):
#    width, height = 100, 100
    image = Image.new('RGB', (width, height), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    byte_arr = BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return byte_arr

class ImageTasks(TaskSet):
    # @task
    # def upload_image(self):
    #     image = generate_random_image(100, 100)
    #     files = {'file': ('test.jpg', image, 'image/jpeg')}
    #     self.client.post("/upload/", files=files)
    #     self.wait()
    
    # @task
    # def search_similar(self):
    #     client = weaviate.Client(url=WEAVIATE_URL)
    #     try:
    #         # 무작위로 저장된 벡터 선택
    #         query_result = client.query.get("ImageEmbedding", ["embedding"]).with_limit(1).do()
    #         if query_result and query_result['data']['Get']['ImageEmbedding']:
    #             random_vector = query_result['data']['Get']['ImageEmbedding'][0]['embedding']
    #         else:
    #             raise Exception("No embeddings found in Weaviate")

    #         start_time = time.time()
    #         search_result = client.query.get("ImageEmbedding", ["uuid", "_additional {certainty}"])\
    #             .with_near_vector({"vector": random_vector})\
    #             .with_limit(5)\
    #             .do()
    #         end_time = time.time()

    #         search_time = end_time - start_time
    #         print(f"Search time: {search_time:.4f} seconds")

    #         for result in search_result['data']['Get']['ImageEmbedding']:
    #             certainty = result['_additional']['certainty']
    #             cosine_similarity = weaviate_certainty_to_cosine(certainty)
    #             print(f"UUID: {result['uuid']}, Cosine Similarity: {cosine_similarity:.4f}")

    #     except Exception as e:
    #         print(f"Error during search_similar task: {e}")

    #     self.wait()


    @task
    def search_similar(self):
        image = generate_random_image(100, 100)
        files = {'file': ('test.jpg', image, 'image/jpeg')}
        self.client.post("/search_similar/", files=files) 
        self.wait()

        
#    @task
#    def upload_large_image(self):
#        num_images = random.randint(10, 1000)
#        for _ in range(num_images):
#            # 이미지 크기를 설정합니다.
#            image = generate_random_image(3000, 3000)   # 100, 1000, 3000
#            files = {'file': ('test.jpg', image, 'image/jpeg')}
#            self.client.post("/upload/", files=files)
#            self.wait()

#    @task
#    def upload_multiple_images(self):
#        num_images = random.randint(10, 1000)  # 각 클라이언트가 업로드할 이미지 수를 랜덤으로 설정
#        for _ in range(num_images):
#            image = generate_random_image()
#            files = {'file': ('test.jpg', image, 'image/jpeg')}
#            self.client.post("/upload/", files=files)
#            self.wait()

        

class WebsiteUser(HttpUser):
    tasks = [ImageTasks]
    wait_time = between(1, 5)
    min_wait = 5000  # 최소 대기 시간
    max_wait = 15000  # 최대 대기 시간