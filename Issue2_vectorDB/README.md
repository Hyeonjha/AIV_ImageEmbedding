# PostgreSQL Docker 이미지 실행
```terminal
docker run --name my_postgres -e POSTGRES_PASSWORD=aiv11011 -d -p 5432:5432 postgres
```

# 가상환경 생성, 활성화

```terminal
 python -m venv venv
 source venv/bin/activate
```

# 가상 환경 활성화된 상태에서 필요한 패키지 설치
```terminal
pip install psycopg2-binary sqlalchemy torch torchvision pillow numpy
```

# PostgreSQL 셸에서 데이터 확인
# 1. PostgreSQL 셸 접속
# 2. 데이터베이스 ㅡ사용
# 3. 데이터 조회
```terminal
docker exec -it my_postgres psql -U postgres
\c postgres
SELECT * FROM image_embeddings;
```

# 테이블 삭제
```terminal
DROP TABLE IF EXISTS image_embeddings;
```

# 테이블 다시 생성
```terminal
CREATE TABLE image_embeddings (
    id SERIAL PRIMARY KEY,
    image_path VARCHAR(255) UNIQUE NOT NULL,
    label VARCHAR(255) NOT NULL,
    embedding FLOAT[] NOT NULL
);
```
### Milvus 설치 및 설정 

# 1. Milvus 실행 위해 `docker-compose.yml` 파일 작성
#    해당 파일 있는 디렉토리에서 다음 명령어 실행 -> Milvus 시행
```terminal
docker-compose up -d
```

# 2. PyMilvus 설치
#    Milvus와 상호작용하기 위해 Python 클라이언트 라이브러리인 `pymilvus` 설치
```terminal
pip install pymilvus
```




# 설치 스크립트 다운로드
```terminal
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
```

# 스크립트 실행 권한 부여
```terminal
chmod +x standalone_embed.sh
```

# Docker 컨테이너 시작
```terminal
bash standalone_embed.sh start
```

# Milvus 컨테이너 상태 확인 - 컨테이너가 정상적으로 실행 중인지 확인하려면 다음 명령어를 사용합니다.
```terminal
docker ps
```
#   실행 중인 컨테이너 목록에 milvus-standalone 컨테이너가 표시, STATUS가 healthy로 표시되면 정상적으로 실행된 것


# 의존성 install
```terminal
pip install -r requirements.txt
```






#
```terminal
docker run -d --name weaviate --env QUERY_DEFAULTS_LIMIT=25 --env CLUSTERS_PEERS_ACTION_BATCHING_WORKERS=4 --env CLUSTER_SLAVES_COUNT=2 --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true --restart=unless-stopped --env PERSISTENCE_DATA_PATH="/var/lib/weaviate" --volume /var/lib/weaviate:/var/lib/weaviate --publish 8080:8080 semitechnologies/weaviate:1.19.2
```
