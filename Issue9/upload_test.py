# /Kafka_Final/test/upload_test.py
import requests
import os

# API 엔드포인트
upload_url = "http://localhost:5001/upload/"

# 테스트 이미지 디렉토리
test_image_dir = "/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/train_L"

# 이미지 업로드 함수
def upload_image(file_path):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(upload_url, files=files)
        return response.json()

# 디렉토리 내 모든 파일을 재귀적으로 찾는 함수
def find_all_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

# 테스트 이미지 업로드
for image_file in find_all_files(test_image_dir):
    response = upload_image(image_file)
    print(f"Uploaded {os.path.basename(image_file)}: {response}")

