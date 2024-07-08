from locust import HttpUser, TaskSet, task, between
from io import BytesIO
from PIL import Image
import random

def generate_random_image(width, height):
#    width, height = 100, 100
    image = Image.new('RGB', (width, height), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    byte_arr = BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return byte_arr

class ImageTasks(TaskSet):
#    @task(10)
#    def upload_image(self):
#        """
#        image = generate_random_image()
#        files = {'file': ('test.jpg', image, 'image/jpeg')}
#        self.client.post("/upload/", files=files)
#        """
#        pass  # 비활성화
        
    @task
    def upload_large_image(self):
        num_images = random.randint(10, 1000)
        for _ in range(num_images):
            # 이미지 크기를 설정합니다.
            image = generate_random_image(3000, 3000)   # 100, 1000, 3000
            files = {'file': ('test.jpg', image, 'image/jpeg')}
            self.client.post("/upload/", files=files)
            self.wait()

#    @task
#    def upload_multiple_images(self):
#        num_images = random.randint(10, 1000)  # 각 클라이언트가 업로드할 이미지 수를 랜덤으로 설정
#        for _ in range(num_images):
#            image = generate_random_image()
#            files = {'file': ('test.jpg', image, 'image/jpeg')}
#            self.client.post("/upload/", files=files)
#            self.wait()


#    @task(1)
#    def search_similar(self):
#        """
#        image = generate_random_image()
#        files = {'file': ('test.jpg', image, 'image/jpeg')}
#        self.client.post("/search_similar/", files=files)   
#        """
#        pass  # 비활성화

    def wait(self):
        self.user.wait()
        

class WebsiteUser(HttpUser):
    tasks = [ImageTasks]
    wait_time = between(1, 5)
