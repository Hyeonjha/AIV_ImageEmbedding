FROM locustio/locust

# Pillow 설치
RUN pip install pillow

# numpy 설치
RUN pip install numpy

# 작업 디렉토리 설정
WORKDIR /mnt/locust

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Locust 스크립트를 복사
COPY locustfile.py /mnt/locust/locustfile.py

CMD ["locust"]
