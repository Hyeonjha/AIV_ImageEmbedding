FROM python:3.8-slim

WORKDIR /app

# 애플리케이션 소스 코드 복사
COPY . /app

# 필요한 패키지 설치
RUN pip install -r requirements.txt

# wait-for-it.sh 스크립트를 복사하고 실행 권한 부여
COPY wait-for-it.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/wait-for-it.sh

# 컨테이너 시작 시 초기화 스크립트를 실행하고 앱을 실행
CMD ["wait-for-it.sh", "kafka:9092", "--", "wait-for-it.sh", "weaviate:8080", "--", "python", "main.py"]
