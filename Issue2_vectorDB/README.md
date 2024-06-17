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
└── requirements.txt
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

### 3. Docker Compose 빌드 및 실행

Docker Compose를 사용하여 PostgreSQL 컨테이너를 빌드하고 실행합니다.

```sh
docker-compose up -d --build
```

### 4. 데이터베이스 초기화 확인

PostgreSQL 컨테이너가 정상적으로 실행되고 초기화 스크립트가 실행되었는지 확인합니다. 로그를 확인하여 초기화 스크립트가 성공적으로 실행되었는지 확인합니다.

```sh
docker logs <컨테이너_ID>
```

### 5. 이미지 임베딩 저장

`postgresStore.py` 스크립트를 실행하여 이미지 임베딩을 데이터베이스에 저장합니다. 이 스크립트는 `data-gatter/train_L` 디렉토리의 이미지를 처리합니다.

```sh
python postgresStore.py
```

### 6. 유사한 이미지 검색

`postgresFind.py` 스크립트를 실행하여 유사한 이미지를 검색합니다. 이 스크립트는 `data-gatter/testcopy/bubble_381007.jpg` 이미지를 기준으로 유사한 이미지를 찾습니다. (변경 가능)

```sh
python postgresFind.py
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

### requirements.txt

프로젝트에 필요한 Python 패키지 목록.
