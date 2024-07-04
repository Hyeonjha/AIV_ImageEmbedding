# 이미지 업로드 및 임베딩 서비스

이 프로젝트는 이미지를 업로드(PostgreSQL)하고, 이미지를 Kafka를 통해 전달하며, 임베딩을 생성하고 Weaviate에 저장하는 기능을 제공합니다.

## 프로젝트 구조
```lua
/Kafka_Final
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
|-- embedding-worker-2/
|   |-- Dockerfile
|   |-- embedding_service.py
|   |-- requirements.txt
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
