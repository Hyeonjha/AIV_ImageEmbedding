# 이미지 업로드 및 임베딩 서비스

이 프로젝트는 이미지를 업로드(PostgreSQL)하고, 이미지를 Kafka를 통해 전달하며, 임베딩을 생성하고 Weaviate에 저장하는 기능을 제공합니다.

## 프로젝트 구조
```lua
/Issue9
|-- docker-compose.yml
|-- requirements.txt
|-- image-service/
|   |-- Dockerfile
|   |-- main.py
|   |-- requirements.txt
|-- embedding-worker-1/
|   |-- Dockerfile
|   |-- embedding_service.py
|   |-- requirements.txt
|-- locust/
|   |-- Dockerfile
|   |-- locustfile.py
|-- upload_test.py
|-- README.md
```

## 설치 및 실행

### 1. Docker Compose 설정
`docker-compose.yml` 파일은 각 서비스 (PostgreSQL, Kafka, FastAPI 등)를 설정합니다. 각 서비스가 네트워크 `app_net`을 통해 서로 통신할 수 있도록 설정되어 있습니다.

### 2. 이미지 서비스 (image-service)
FastAPI를 사용하여 이미지를 업로드하고, 이미지 메타데이터를 PostgreSQL에 저장하며, Kafka를 통해 UUID를 전달합니다.

#### 주요 코드
- **image-service/Dockerfile**: FastAPI 애플리케이션을 위한 Docker 이미지를 빌드합니다.
- **image-service/main.py**: FastAPI 엔드포인트를 정의하고, 이미지를 저장하며, 메타데이터를 PostgreSQL에 저장하고, Kafka에 메시지를 보냅니다.

### 3. 임베딩 서비스 (embedding-service)
Kafka에서 메시지를 소비하고, 이미지를 임베딩하여 Weaviate에 저장합니다.

#### 주요 코드
- **embedding-worker-1/Dockerfile**: 임베딩 서비스 애플리케이션을 위한 Docker 이미지를 빌드합니다.
- **embedding-worker-1/embedding_service.py**: Kafka에서 이미지를 소비하고, 임베딩을 생성하여 Weaviate에 저장합니다.
- **embedding-worker-2/Dockerfile**: 임베딩 서비스 애플리케이션을 위한 Docker 이미지를 빌드합니다.
- **embedding-worker-2/embedding_service.py**: Kafka에서 이미지를 소비하고, 임베딩을 생성하여 Weaviate에 저장합니다.

### 4. Locust를 이용한 부하 테스트 (locust)
Locust를 사용하여 이미지 업로드 및 유사 이미지 검색 API의 부하 테스트를 수행합니다.

#### 주요 코드
- **locust/Dockerfile**: Locust 애플리케이션을 위한 Docker 이미지를 빌드합니다.
- **locust/locustfile.py**: Locust 스크립트로, 이미지 업로드 및 유사 이미지 검색 작업을 정의합니다.

### 5. 테스트 스크립트 (upload_test.py)
test 이미지 dataset 업로드를 위한 스크립트입니다.



## 실행 방법
프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 Docker Compose로 모든 서비스를 시작합니다:
```sh
docker-compose up --build
```

### 1. Swagger UI 접속 및 이미지 업로드 테스트
브라우저에서 `http://localhost:5001/docs`로 접속하여 `/upload/` 엔드포인트를 사용하여 이미지를 업로드합니다.

### 2. Kafka UI에서 메시지 확인
브라우저에서 `http://localhost:8081`로 접속하여 `image_topic`에 메시지가 들어왔는지 확인합니다.

### 3. Weaviate에서 임베딩 데이터 확인
브라우저에서 `http://localhost:8080/v1/objects`로 접속하여 저장된 이미지 임베딩 데이터를 확인합니다.

### 4. Locust를 이용한 부하 테스트 실행
브라우저에서 `http://localhost:8089`로 접속하여 Locust 부하 테스트를 시작합니다.

### (필요 시) 테스트 스크립트 실행
다음 명령어를 실행하여 `upload_test.py` 스크립트를 실행합니다:
```sh
python3 /path/to/upload_test.py
```


## locust 결과 정리

### 1. upload test
```
Issue9/locustResult/uploadImg.html
```

### 2. similar search test (303개 저장된 상태에서 검색) 
```
Issue9/locustResult/searchSim.html
```

### 3.  여러 client가 동시에 수십 수백장을 업로드 요청 테스트
```
Issue9/locustResult/multiUpload.html
```

### 4. 이미지 사이즈 1000 * 1000
```
Issue9/locustResult/imgSize10.html
```

### 4-1. 이미지 사이즈 1000 * 1000 + 한번에 수십, 수백장 업로드
```
Issue9/locustResult/imgSize10_multi.html
```

### 5. 이미지 사이즈 3000 * 3000 + 한번에 수십, 수백장 업로드
```
Issue9/locustResult/imgSize30_multi.html
```





## 7/10 결과 재정리
### 1. 가지고 있던 test data 업로드, 검색
```
Issue9/locustResult/searchSim_.html
```

```
embed_image took 0.45265913009643555 seconds
search_similar_in_weaviate took 0.0861973762512207 seconds

embed_image took 0.6620943546295166 seconds
search_similar_in_weaviate took 0.0813436508178711 seconds
```

### 2. 사이즈 100*100 업로드(약 500개 업로드), 검색
```
Issue9/locustResult/upConv100.html
Issue9/locustResult/searchSim_Conv100.html
```

```
read_image took 2.6226043701171875e-06 seconds
save_image took 0.00022029876708984375 seconds
save_to_db took 0.011336088180541992 seconds
send_to_kafka took 0.001672983169555664 seconds

read_image took 3.814697265625e-06 seconds
save_image took 0.0002071857452392578 seconds
save_to_db took 0.010311365127563477 seconds
send_to_kafka took 0.0002906322479248047 seconds

read_image took 4.291534423828125e-06 seconds
save_image took 0.001852273941040039 seconds
save_to_db took 0.021559953689575195 seconds
send_to_kafka took 0.0002925395965576172 seconds

read_image took 2.86102294921875e-06 seconds
save_image took 0.001077413558959961 seconds
save_to_db took 0.010017633438110352 seconds
send_to_kafka took 0.0006411075592041016 seconds
```
```
preprocess_resize took 0.0010445117950439453 seconds
preprocess_center_crop took 0.0001304149627685547 seconds
preprocess_to_tensor took 0.0011560916900634766 seconds
preprocess_normalize took 0.00042176246643066406 seconds
embed_image took 0.6649417877197266 seconds
search_similar_in_weaviate took 0.1017451286315918 seconds

preprocess_resize took 0.0012297630310058594 seconds
preprocess_center_crop took 0.00019049644470214844 seconds
preprocess_to_tensor took 0.002794504165649414 seconds
preprocess_normalize took 0.0034360885620117188 seconds
embed_image took 0.4213104248046875 seconds
search_similar_in_weaviate took 0.06925463676452637 seconds
```

### 3. 사이즈 1000*1000 업로드(약 500개 업로드), 검색
```
Issue9/locustResult/upConv1000.html
Issue9/locustResult/searchSim_Conv1000.html
```

```
read_image took 2.1457672119140625e-06 seconds
save_image took 0.0002739429473876953 seconds
save_to_db took 0.007207393646240234 seconds
send_to_kafka took 0.0003101825714111328 seconds

read_image took 4.0531158447265625e-06 seconds
save_image took 0.00034165382385253906 seconds
save_to_db took 0.015622615814208984 seconds
send_to_kafka took 0.0016965866088867188 seconds
```

```
preprocess_resize took 0.010740280151367188 seconds
preprocess_center_crop took 0.0002155303955078125 seconds
preprocess_to_tensor took 0.0028815269470214844 seconds
preprocess_normalize took 0.00724029541015625 seconds
embed_image took 0.4384312629699707 seconds
search_similar_in_weaviate took 0.123016357421875 seconds

preprocess_resize took 0.009558916091918945 seconds
preprocess_center_crop took 0.00018858909606933594 seconds
preprocess_to_tensor took 0.0009210109710693359 seconds
preprocess_normalize took 0.0003294944763183594 seconds
embed_image took 0.4189567565917969 seconds
search_similar_in_weaviate took 0.0811319351196289 seconds

preprocess_resize took 0.009733438491821289 seconds
preprocess_center_crop took 0.00020074844360351562 seconds
preprocess_to_tensor took 0.0009264945983886719 seconds
preprocess_normalize took 0.0004818439483642578 seconds
embed_image took 0.39894556999206543 seconds
search_similar_in_weaviate took 0.13485097885131836 seconds
```

### 4. 사이즈 3000*3000 업로드(약 500개 업로드), 검색
```
Issue9/locustResult/upConv3000.html
Issue9/locustResult/searchSim_Conv3000.html
```

```
read_image took 3.0994415283203125e-06 seconds
save_image took 0.001293182373046875 seconds
save_to_db took 0.005756378173828125 seconds
send_to_kafka took 0.00025153160095214844 seconds

read_image took 3.0994415283203125e-06 seconds
save_image took 0.001285552978515625 seconds
save_to_db took 0.007987737655639648 seconds
send_to_kafka took 0.00030493736267089844 seconds

read_image took 2.6226043701171875e-06 seconds
save_image took 0.004534482955932617 seconds
save_to_db took 0.006303310394287109 seconds
send_to_kafka took 0.00025200843811035156 seconds
```

```

```

** embed_image에서 조금 오래 걸림 **
ConvNeXt -> ResNet18 or ResNet 50 사용해보기