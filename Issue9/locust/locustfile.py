from locust import HttpUser, TaskSet, task, between
import os
import uuid
from io import BytesIO
from PIL import Image
import random

def generate_random_image():
    width, height = 100, 100
    image = Image.new('RGB', (width, height), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    byte_arr = BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return byte_arr

class ImageTasks(TaskSet):
    @task(1)
    def upload_image(self):
        image = generate_random_image()
        files = {'file': ('test.jpg', image, 'image/jpeg')}
        self.client.post("/upload/", files=files)

    @task(2)
    def search_similar(self):
        image = generate_random_image()
        files = {'file': ('test.jpg', image, 'image/jpeg')}
        self.client.post("/search_similar/", files=files)

class WebsiteUser(HttpUser):
    tasks = [ImageTasks]
    wait_time = between(1, 5)
