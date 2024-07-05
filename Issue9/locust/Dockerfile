FROM locustio/locust

# Pillow 설치
RUN pip install pillow

# 작업 디렉토리 설정
WORKDIR /mnt/locust

# Locust 스크립트를 복사
COPY locustfile.py /mnt/locust/locustfile.py

CMD ["locust"]
