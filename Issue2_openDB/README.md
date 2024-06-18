# Issue2_openDB 프로젝트
Weaviate 데이터베이스를 사용하여 이미지 임베딩을 저장하고 유사한 이미지를 검색하는 기능 구현. Docker를 사용하여 Weaviate를 설정하고, Python 스크립트를 통해 이미지 임베딩을 처리.

## 디렉토리 구조
```
Issue2_VectorDB/
├── docker-compose.yml
├── Img2Vec.py
├── initializeDB.py
├── weaviateStore.py
├── weaviateFind.py
├── weaviatePerf.py
├── weaviateSim.py
├── weavInsertV.py
├── weavSearchT.py
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

### initializeDB.py

데이터베이스에 저장된 데이터 초기화.

### weavInsertV.py

fake vector 생성하여 10만개의 데이터 저장하는 모델의 성능 평가

### weavSearchT.py

10만개의 데이터 검색하는 모델의 성능 평가

### requirements.txt

프로젝트에 필요한 Python 패키지 목록.

## 데이터 폴더

- **data-gatter/train_L**: 학습용 이미지가 포함된 폴더. 이미지는 레이블별로 하위 폴더에 정리되어 있어야 합니다.
- **data-gatter/test**: 검색 기능을 테스트하기 위한 이미지가 포함된 폴더.



## 성능 평가 

### 임베딩 저장, 검색 성능 평가 실행 (10,000개) 

출력 결과:

```
Insert - Mean time: 0.008721607756614685, Std time: 0.0015508988667878847
Search - Mean time: 0.013982857489585877, Std time: 0.001693721139744062
```

### 임베딩 저장, 검색 성능 평가 실행 (10만개의 fake vector 생성 후 저장, 검색 -> 10,000개 단위)

```sh
python weavInsertV.py
```

출력 결과:

```
Inserted 10000 vectors - Batch mean time: 0.0090, Batch std time: 0.0016
Inserted 20000 vectors - Batch mean time: 0.0136, Batch std time: 0.0032
Inserted 30000 vectors - Batch mean time: 0.0164, Batch std time: 0.0034
Inserted 40000 vectors - Batch mean time: 0.0181, Batch std time: 0.0042
Inserted 50000 vectors - Batch mean time: 0.0199, Batch std time: 0.0051
Inserted 60000 vectors - Batch mean time: 0.0215, Batch std time: 0.0057
Inserted 70000 vectors - Batch mean time: 0.0226, Batch std time: 0.0061
Inserted 80000 vectors - Batch mean time: 0.0234, Batch std time: 0.0061
Inserted 90000 vectors - Batch mean time: 0.0256, Batch std time: 0.0076
Inserted 100000 vectors - Batch mean time: 0.0284, Batch std time: 0.0131
Total Insert - Mean time: 0.0198, Std time: 0.0084
```

```sh
python weavSearchT.py

출력 결과:

```
Searched 10000 vectors - Batch mean time: 0.0179, Batch std time: 0.0018
Searched 20000 vectors - Batch mean time: 0.0179, Batch std time: 0.0022
Searched 30000 vectors - Batch mean time: 0.0177, Batch std time: 0.0032
Searched 40000 vectors - Batch mean time: 0.0169, Batch std time: 0.0012
Searched 50000 vectors - Batch mean time: 0.0168, Batch std time: 0.0012
Searched 60000 vectors - Batch mean time: 0.0169, Batch std time: 0.0017
Searched 70000 vectors - Batch mean time: 0.0168, Batch std time: 0.0012
Searched 80000 vectors - Batch mean time: 0.0171, Batch std time: 0.0019
Searched 90000 vectors - Batch mean time: 0.0170, Batch std time: 0.0025
Searched 100000 vectors - Batch mean time: 0.0169, Batch std time: 0.0016
Total Search - Mean time: 0.0172, Std time: 0.0020
```