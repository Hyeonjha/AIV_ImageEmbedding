# Issue2_VectorDB 프로젝트
Weaviate 데이터베이스를 사용하여 이미지 임베딩을 저장하고 유사한 이미지를 검색하는 기능 구현. Docker를 사용하여 Weaviate를 설정하고, Python 스크립트를 통해 이미지 임베딩을 처리.

## 디렉토리 구조
```
Issue2_VectorDB/
├── docker-compose.yml
├── Img2Vec.py
├── weaviateStore.py
├── weaviateFind.py
├── weaviatePerf.py
├── weaviateSim.py
├── requirements.txt
└── README.md
```

### Python 설정

1. 가상 환경을 생성하고 활성화합니다:

   ```sh
   python -m venv venv
   source venv/bin/activate  # Windows에서는 `venv\Scripts\activate` 사용
   ```

2. 필요한 패키지를 설치합니다:

   ```sh
   pip install -r requirements.txt
   ```

## 설치 및 실행 방법

### 1. Docker 및 Docker Compose 설치

Docker와 Docker Compose가 설치되어 있어야 합니다. 설치되지 않았다면 다음 링크를 참조하여 설치합니다.
- [Docker 설치](https://docs.docker.com/get-docker/)
- [Docker Compose 설치](https://docs.docker.com/compose/install/)

### 2. 프로젝트 클론 및 이동

프로젝트를 클론하고 해당 디렉토리로 이동합니다.

```sh
git clone <프로젝트_저장소_URL>
cd Issue2_VectorDB
```

### 3. Docker Compose를 사용하여 Weaviate 컨테이너 빌드 및 실행

Docker Compose를 사용하여 Weaviate 컨테이너를 빌드하고 실행합니다.

```sh
docker-compose up -d --build
```

### 4. Weaviate 클라이언트 확인

Weaviate 컨테이너가 정상적으로 실행되고 있는지 확인합니다. 로그를 확인하여 Weaviate 서버가 정상적으로 실행되었는지 확인합니다.

```sh
docker logs <컨테이너_ID>
```

## 스크립트 실행

### 1. 이미지 임베딩 저장

`weaviateStore.py` 스크립트를 실행하여 이미지 임베딩을 데이터베이스에 저장합니다. 이 스크립트는 `data-gatter/train_L` 디렉토리의 이미지를 처리합니다.

```sh
python weaviateStore.py
```

### 2. 유사한 이미지 검색

`weaviateFind.py` 스크립트를 실행하여 유사한 이미지를 검색합니다. 이 스크립트는 `data-gatter/test_images` 디렉토리의 이미지를 기준으로 유사한 이미지를 찾습니다.

```sh
python weaviateFind.py
```

### 3. 삽입 및 검색 성능 평가

#### 이미지 임베딩 저장 성능 평가

`weaviatePerf.py` 스크립트를 실행하여 이미지 임베딩을 데이터베이스에 저장하는 성능을 평가합니다. 이 스크립트는 10,000회 반복하여 삽입 시간의 평균 및 표준 편차를 측정하고 출력합니다.

```sh
python weaviatePerf.py
```

#### 유사한 이미지 검색 성능 평가

`weaviatePerf.py` 스크립트를 실행하여 데이터베이스에서 유사한 이미지를 검색하는 성능을 평가합니다. 이 스크립트는 10,000회 반복하여 검색 시간의 평균 및 표준 편차를 측정하고 출력합니다.

```sh
python weaviatePerf.py
```

### 4. 유사도 계산 비교

`weaviateSim.py` 스크립트를 실행하여 Weaviate에서 계산한 유사도 값과 수동으로 계산한 코사인 유사도 값을 비교하여 일치하는지 확인합니다.

```sh
python weaviateSim.py
```

## 파일 설명

### docker-compose.yml

Docker Compose 설정 파일. Weaviate 컨테이너 설정.

### Img2Vec.py

이미지 임베딩 생성하는 클래스 정의 파일. ConvNeXT 모델 사용하여 이미지를 임베딩 벡터로 변환.

### weaviateStore.py

이미지를 임베딩하여 Weaviate 데이터베이스에 저장하는 스크립트.

### weaviateFind.py

Weaviate 데이터베이스에서 유사한 이미지를 검색하는 스크립트.

### weaviatePerf.py

이미지 임베딩을 Weaviate 데이터베이스에 저장하고 검색 성능을 측정하는 스크립트.

### weaviateSim.py

Weaviate에서 계산한 유사도 값과 수동으로 계산한 코사인 유사도 값을 비교하는 스크립트.

### requirements.txt

프로젝트에 필요한 Python 패키지 목록.

## 데이터 폴더

- **data-gatter/train_L**: 학습용 이미지가 포함된 폴더. 이미지는 레이블별로 하위 폴더에 정리되어 있어야 합니다.
- **data-gatter/test**: 검색 기능을 테스트하기 위한 이미지가 포함된 폴더.


## 예제 (20회 반복했을 때의 결과)

### 이미지 임베딩 저장, 검색 성능 평가 실행

```sh
python weaviatePerf.py
```

출력 예시:

```
Insert - Mean time: 0.008717226982116699, Std time: 0.0009743874480479035
Search - Mean time: 0.013183975219726562, Std time: 0.0018419031956134915
```

### 유사도 계산 비교 실행

```sh
python weaviateSim.py
```

출력 예시:

```
Mean difference between Weaviate and manual cosine similarity: 2.3638725029200028e-07
Standard deviation of difference: 1.8026255796318865e-07
```