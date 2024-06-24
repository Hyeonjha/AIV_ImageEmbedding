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
├── postgreSim.py
├── postInsertV.py
├── postSearchT.py
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

### 1-1. 성능 평가와 함께 가짜 벡터 저장 

이 스크립트는 100,000개의 가짜 벡터를 생성하여 PostgreSQL 데이터베이스에 저장합니다. 또한, 이 데이터들의 삽입 시간의 평균 및 표준 편차를 10,000회마다 측정하고 출력합니다.

```sh
python postInsertV.py
```

### 2. 유사한 이미지 검색

`postgresFind.py` 스크립트를 실행하여 유사한 이미지를 검색합니다. 이 스크립트는 `data-gatter/testcopy/bubble_381007.jpg` 이미지를 기준으로 유사한 이미지를 찾습니다. (변경 가능)

```sh
python postgresFind.py
```

### 2-1. 성능 평가와 함께 벡터 검색

이 스크립트는 PostgreSQL 데이터베이스에서 코사인 유사도를 사용하여 유사한 이미지를 검색합니다. 100,000회 반복하여 검색 시간의 평균 및 표준 편차를 10,000회마다 측정하고 출력합니다.

```sh
python postSearchT.py
```

### 3. 유사도 계산 비교

이 스크립트는 PostgreSQL에서 계산한 유사도 값과 수동으로 계산한 코사인 유사도 값을 비교하여 일치하는지 확인합니다.

```sh
python postgreSim.py
```

### 4. 검색 속도 개선 시도
GIST -> LSH -> HNSW(별도 인덱스파일 저장 방식) -> HNSW(데이터베이스 내부에 저장, 인덱스 변환 필요) -> DB 제공 index(ivfflat, hnsw) 
=> 성능 개선 X


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

### postgreSim.py

PostgreSQL에서 계산한 유사도 값과 수동으로 계산한 코사인 유사도 값을 비교하는 스크립트.

### postInsertV.py

fake vector 생성하여 10만개의 벡터를 데이터베이스에 저장하고 삽입 성능을 측정하는 스크립트.

### postSearchT.py

10만개의 데이터 검색하고 검색 성능을 측정하는 스크립트.

### requirements.txt

프로젝트에 필요한 Python 패키지 목록.


## 데이터 폴더

- **data-gatter/train_L**: 학습용 이미지가 포함된 폴더. 이미지는 레이블별로 하위 폴더에 정리되어 있어야 합니다.
- **data-gatter/test**: 검색 기능을 테스트하기 위한 이미지가 포함된 폴더.


## 성능 평가

### 임베딩 저장, 검색 성능 평가 실행 (10,000개 저장, 검색) 

출력 결과:

```
Insert times over 10000 embeddings: 0.009964231491088868 ± 0.0015895360776028442 seconds

Search times over 10000 iterations: 0.05208480780124664 ± 0.004131978767582321 seconds
```

### 임베딩 저장 성능 평가 실행 (10만개의 fake vector 생성 후 저장 -> 10,000개 단위)

```sh
python postInsertV.py
```

출력 결과:

```
Inserting 10000 embeddings...
Insert times over 10000 embeddings: 0.00987683801651001 ± 0.0034335988844233934 seconds
Inserting 20000 embeddings...
Insert times over 10000 embeddings: 0.00980962405204773 ± 0.002016624127072981 seconds
Inserting 30000 embeddings...
Insert times over 10000 embeddings: 0.009833124709129334 ± 0.002312757151848678 seconds
Inserting 40000 embeddings...
Insert times over 10000 embeddings: 0.010633320093154907 ± 0.003667472597662235 seconds
Inserting 50000 embeddings...
Insert times over 10000 embeddings: 0.010132505178451538 ± 0.0019775696195001567 seconds
Inserting 60000 embeddings...
Insert times over 10000 embeddings: 0.00995038526058197 ± 0.002248606919583813 seconds
Inserting 70000 embeddings...
Insert times over 10000 embeddings: 0.01005813102722168 ± 0.0023467888584746592 seconds
Inserting 80000 embeddings...
Insert times over 10000 embeddings: 0.009985102987289428 ± 0.0018872216636339483 seconds
Inserting 90000 embeddings...
Insert times over 10000 embeddings: 0.009961880087852478 ± 0.002118598653167788 seconds
Inserting 100000 embeddings...
Insert times over 10000 embeddings: 0.01068202040195465 ± 0.0037268060481944892 seconds
```

### 검색 성능 평가 실행 (10만개의 fake vector 저장된 상태에서 검색 -> 10,000개 단위)

```sh
python postSearchT.py
```

출력 결과:

100,000개의 데이터베이스가 저장된 상태에서의 성능평가 -> 데이터베이스에 데이터 10,000개 저장되어있을때보다 검색 시간 16배 증가
```
Search times for 100 queries: 0.8379068636894226 ± 0.09632197552364523 seconds

Search times for 1000 queries: 0.8289583191871643 ± 0.09643168655288632 seconds
```

### 데이터 저장 방식 변경 - GIST
```
Inserting 10000 embeddings...
Insert times over 10000 embeddings: 0.014438557243347168 ± 0.013971510376183268 seconds

10만개 저장 후 검색
Search times for 1000 queries: 0.05896414399147034 ± 0.01322378063261405 seconds
```

### LSH(Locality-Sensitive Hashing) -->> 실제 이미지 임베딩 후 검색 시 error

10만개 저장 결과 : 
```
Inserting 10000 embeddings...
Insert times over 10000 embeddings: 0.011870080924034118 ± 0.0025198901308637685 seconds
Inserting 20000 embeddings...
Insert times over 10000 embeddings: 0.012624620509147644 ± 0.0032029123330840595 seconds
Inserting 30000 embeddings...
Insert times over 10000 embeddings: 0.024018246960639953 ± 1.0248020795409534 seconds
Inserting 40000 embeddings...
Insert times over 10000 embeddings: 0.011922398781776429 ± 0.012237047499576322 seconds
...
```

10만개 저장 후 검색(fake vector 평가 시) -> 실제 이미지 임베딩 error
```
Search times for 10000 queries: 0.01549175112247467 ± 0.0012841886448967997 seconds
Search times for 20000 queries: 0.016615263557434083 ± 0.006843545815597204 seconds
Search times for 30000 queries: 0.015955161428451537 ± 0.002753508182131156 seconds
Search times for 40000 queries: 0.015576725387573242 ± 0.0010069969501449879 seconds
...
```

### HNSW
검색 속도 굉장히 빠름 but 별도 파일 관리 필요
```
Inserting 10000 embeddings...
Insert times over 10000 embeddings: 0.009714061450958251 ± 0.0028533289559459383 seconds
Inserting 20000 embeddings...
Insert times over 10000 embeddings: 0.009527180123329163 ± 0.0016754687110578984 seconds
Inserting 30000 embeddings...
Insert times over 10000 embeddings: 0.009725466609001159 ± 0.002710057932984796 seconds
Inserting 40000 embeddings...
Insert times over 10000 embeddings: 0.009865274024009704 ± 0.0024887662027976553 seconds
Inserting 50000 embeddings...
Insert times over 10000 embeddings: 0.009631681990623473 ± 0.0020928812317590017 seconds
Inserting 60000 embeddings...
Insert times over 10000 embeddings: 0.009735783648490906 ± 0.0021875225940249645 seconds
Inserting 70000 embeddings...
Insert times over 10000 embeddings: 0.00970157196521759 ± 0.003418452778373024 seconds
Inserting 80000 embeddings...
Insert times over 10000 embeddings: 0.010224883794784546 ± 0.0032901747785833713 seconds
Inserting 90000 embeddings...
Insert times over 10000 embeddings: 0.009743905162811279 ± 0.002305780842385575 seconds
Inserting 100000 embeddings...
Insert times over 10000 embeddings: 0.009627681875228881 ± 0.0027056482820152173 seconds
HNSW index created and saved to 'hnsw_index.bin'
```

```
Search times for batch 10000: 0.00032954392433166504 ± 6.964135275123236e-05 seconds
Search times for batch 20000: 0.00038621418476104735 ± 0.0011958601129773059 seconds
Search times for batch 30000: 0.0003617649078369141 ± 0.0012234454739943092 seconds
Search times for batch 40000: 0.00032183821201324465 ± 8.292580279550775e-05 seconds
Search times for batch 50000: 0.00032471842765808103 ± 5.614353698087352e-05 seconds
Search times for batch 60000: 0.00032751214504241944 ± 6.431411264166197e-05 seconds
Search times for batch 70000: 0.000318647837638855 ± 5.053541921064265e-05 seconds
Search times for batch 80000: 0.0003206585884094238 ± 5.593253236293331e-05 seconds
Search times for batch 90000: 0.00032006371021270753 ± 6.387829651313786e-05 seconds
Search times for batch 100000: 0.0003158594369888306 ± 5.1306217771606445e-05 seconds
```

### ivfflat
10,000개 저장, 검색 :
```
Time taken to save batch 1: 78.52 seconds
Average search time: 0.0565 seconds, Standard deviation: 0.0147 seconds
```

100,000개 저장
```
Time taken to save batch 1: 69.14 seconds
Time taken to save batch 2: 70.56 seconds
Time taken to save batch 3: 82.84 seconds
Time taken to save batch 4: 68.39 seconds
Time taken to save batch 5: 65.14 seconds
Time taken to save batch 6: 72.20 seconds
Time taken to save batch 7: 69.51 seconds
Time taken to save batch 8: 67.38 seconds
Time taken to save batch 9: 76.65 seconds
Time taken to save batch 10: 88.58 seconds
Average save time: 73.04 seconds
Standard deviation of save time: 7.08 seconds
```

100,000개 저장된 상태에서 100개 검색 
```
Batch 1: Average search time: 1.5733 seconds, Standard deviation: 0.0576 seconds
```


### hnsw
```
100,000개 저장
Time taken to save batch 1: 77.0878 seconds
Time taken to save batch 2: 96.9995 seconds
Time taken to save batch 3: 78.9220 seconds
Time taken to save batch 4: 70.8705 seconds
Time taken to save batch 5: 68.4129 seconds
Time taken to save batch 6: 71.0716 seconds
Time taken to save batch 7: 70.2100 seconds
Time taken to save batch 8: 68.4714 seconds
Time taken to save batch 9: 72.7419 seconds
Time taken to save batch 10: 76.0370 seconds
Average save time: 75.08 seconds
Standard deviation of save time: 8.07 seconds


Image Path: fake_path_12777, Label: fake_label_7, Similarity: 0.8000
Image Path: fake_path_60148, Label: fake_label_8, Similarity: 0.8006
Image Path: fake_path_64901, Label: fake_label_1, Similarity: 0.8019
Image Path: fake_path_99209, Label: fake_label_9, Similarity: 0.8008
Image Path: fake_path_43146, Label: fake_label_6, Similarity: 0.8049
Image Path: fake_path_94967, Label: fake_label_7, Similarity: 0.8012
Average search time for 1000 fake embeddings: 0.7079 seconds
Standard deviation of search time: 0.0350 seconds
```