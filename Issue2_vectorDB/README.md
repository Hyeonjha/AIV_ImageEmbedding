# Issue2_vectorDB 프로젝트
PostgreSQL 데이터베이스와 pgvector 확장을 사용하여 이미지 임베딩을 저장하고 유사한 이미지를 검색하는 기능 구현. 
Docker를 사용하여 PostgreSQL을 설정하고, Python 스크립트를 통해 이미지 임베딩을 처리.

## 디렉토리 구조
```
Issue2_vectorDB/
├── docker-compose.yml
├── Dockerfile
├── init-db.sh
├── Img2Vec.py
├── ImageEmbedding.py
├── postgresStore.py
├── postgresFind.py
├── postgresStorePerf.py
├── postgresFindPerf.py
├── postgreSim.py
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
cd Issue2_vectorDB
```

### 3. init-db.sh 실행 권한 설정

`init-db.sh` 파일이 실행 가능 권한을 가지고 있는지 확인합니다.

```sh
chmod +x init-db.sh
```

### 4. 프로젝트 디렉토리로 이동

Docker Compose를 사용하여 PostgreSQL 컨테이너를 빌드하고 실행합니다.

```sh
cd path/to/Issue2_vectorDB
```

### 5. Docker Compose 빌드 및 실행

Docker Compose를 사용하여 PostgreSQL 컨테이너를 빌드하고 실행합니다.

```sh
docker-compose up -d --build
```

### 6. 데이터베이스 초기화 확인

PostgreSQL 컨테이너가 정상적으로 실행되고 초기화 스크립트가 실행되었는지 확인합니다. 로그를 확인하여 초기화 스크립트가 성공적으로 실행되었는지 확인합니다.

```sh
docker logs <컨테이너_ID>
```

## 스크립트 실행

### 1. 이미지 임베딩 저장

`postgresStore.py` 스크립트를 실행하여 이미지 임베딩을 데이터베이스에 저장합니다. 이 스크립트는 `data-gatter/train_L` 디렉토리의 이미지를 처리합니다.

```sh
python postgresStore.py
```

### 1-1. 성능 평가와 함께 이미지 임베딩 저장 (update)

이 스크립트는 ConvNext 모델을 사용하여 이미지 임베딩을 추출하고 PostgreSQL 데이터베이스에 저장합니다. 또한, 10,000회 반복하여 삽입 시간의 평균 및 표준 편차를 측정하고 출력합니다.

```sh
python postgresStorePerf.py
```

### 2. 유사한 이미지 검색

`postgresFind.py` 스크립트를 실행하여 유사한 이미지를 검색합니다. 이 스크립트는 `data-gatter/testcopy/bubble_381007.jpg` 이미지를 기준으로 유사한 이미지를 찾습니다. (변경 가능)

```sh
python postgresFind.py
```

### 2. 성능 평가와 함께 이미지 검색 (update)

이 스크립트는 PostgreSQL 데이터베이스에서 코사인 유사도를 사용하여 유사한 이미지를 검색합니다. 10,000회 반복하여 검색 시간의 평균 및 표준 편차를 측정하고 출력합니다.

```sh
python postgresFindPerf.py
```

### 3. 유사도 계산 비교

이 스크립트는 PostgreSQL에서 계산한 유사도 값과 수동으로 계산한 코사인 유사도 값을 비교하여 일치하는지 확인합니다.

```sh
python postgreSim.py
```

## 파일 설명

### docker-compose.yml

Docker Compose 설정 파일. PostgreSQL 컨테이너 설정.

### Dockerfile

PostgreSQL 이미지에 필요한 패키지 & pgvector 확장을 설치하는 설정 파일.

### init-db.sh

PostgreSQL 데이터베이스 초기화 스크립트. pgvector 확장 설치, 필요한 테이블과 컬럼 생성.

### Img2Vec.py

이미지 임베딩 생성하는 클래스 정의 파일. ConvNext 모델 사용하여 이미지를 임베딩 벡터로 변환.

### ImageEmbedding.py

SQLAlchemy를 사용하여 데이터베이스 테이블을 정의하는 파일. 이미지 경로, 라벨, 임베딩 벡터를 저장.

### postgresStore.py

이미지를 임베딩하여 데이터베이스에 저장하는 스크립트.

### postgresFind.py

데이터베이스에서 유사한 이미지를 검색하는 스크립트.

### postgresStorePerf.py

이미지 임베딩을 데이터베이스에 저장하고 삽입 성능을 측정하는 스크립트.

### postgresFindPerf.py

데이터베이스에서 유사한 이미지를 검색하고 검색 성능을 측정하는 스크립트.

### requirements.txt

프로젝트에 필요한 Python 패키지 목록.


## 데이터 폴더

- **data-gatter/train_L**: 학습용 이미지가 포함된 폴더. 이미지는 레이블별로 하위 폴더에 정리되어 있어야 합니다.
- **data-gatter/test**: 검색 기능을 테스트하기 위한 이미지가 포함된 폴더.


## 예제 (20회 반복했을 때의 결과)

### 이미지 임베딩 저장 성능 평가 실행

```sh
python postgresStorePerf.py
```

출력 예시:

```
Insert times over 20 iterations: 0.013816761970520019 ± 0.004027176911622213 seconds
```

### 이미지 검색 성능 평가 실행

```sh
python postgresFindPerf.py
```

출력 예시:

```
Search times over 20 iterations: 4.3378861784935 ± 0.1205780243903615 seconds
```

### 유사도 계산 비교 실행

```sh
python postgreSim.py
```

출력 예시:

```
All similarities match between PostgreSQL and manually calculated values.
```