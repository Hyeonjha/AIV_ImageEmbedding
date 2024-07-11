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





## ConvNeXt
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

다시 test
Issue9/locustResult/upConvN100.html
Issue9/locustResult/searchSim_ConvN100.html
```

```
read_image took 4.291534423828125e-06 seconds
save_image took 0.001852273941040039 seconds
save_to_db took 0.021559953689575195 seconds
send_to_kafka took 0.0002925395965576172 seconds

read_image took 2.86102294921875e-06 seconds
save_image took 0.001077413558959961 seconds
save_to_db took 0.010017633438110352 seconds
send_to_kafka took 0.0006411075592041016 seconds

다시 test
read_image took 4.291534423828125e-06 seconds
save_image took 0.006590127944946289 seconds
save_to_db took 0.015838623046875 seconds
send_to_kafka took 0.00047659873962402344 seconds

read_image took 3.337860107421875e-06 seconds
save_image took 0.0004134178161621094 seconds
save_to_db took 0.01767587661743164 seconds
send_to_kafka took 0.00034236907958984375 seconds
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

다시 test
preprocess_resize took 0.0012748241424560547 seconds
preprocess_center_crop took 0.00028324127197265625 seconds
preprocess_to_tensor took 0.0012149810791015625 seconds
preprocess_normalize took 0.004419565200805664 seconds
embed_image took 0.5219030380249023 seconds
search_similar_in_weaviate took 0.0705418586730957 seconds

preprocess_resize took 0.0010983943939208984 seconds
preprocess_center_crop took 0.0002658367156982422 seconds
preprocess_to_tensor took 0.001844167709350586 seconds
preprocess_normalize took 0.00038909912109375 seconds
embed_image took 0.364424467086792 seconds
search_similar_in_weaviate took 0.06090688705444336 seconds
```

### 3. 사이즈 1000*1000 업로드(약 500개 업로드), 검색
```
Issue9/locustResult/upConv1000.html
Issue9/locustResult/searchSim_Conv1000.html

다시 test
Issue9/locustResult/upConvN1000.html
Issue9/locustResult/searchSim_ConvN1000.html
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

다시 test
read_image took 3.5762786865234375e-06 seconds
save_image took 0.0017473697662353516 seconds
save_to_db took 0.010738134384155273 seconds
send_to_kafka took 0.0006489753723144531 seconds

read_image took 2.6226043701171875e-06 seconds
save_image took 0.00035572052001953125 seconds
save_to_db took 0.013991832733154297 seconds
send_to_kafka took 0.0007915496826171875 seconds
```

```
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

다시 test
preprocess_resize took 0.011145591735839844 seconds
preprocess_center_crop took 0.0006427764892578125 seconds
preprocess_to_tensor took 0.0054473876953125 seconds
preprocess_normalize took 0.000797271728515625 seconds
embed_image took 0.49765491485595703 seconds
search_similar_in_weaviate took 0.07430243492126465 seconds

preprocess_resize took 0.011011123657226562 seconds
preprocess_center_crop took 0.0003161430358886719 seconds
preprocess_to_tensor took 0.02020406723022461 seconds
preprocess_normalize took 0.021791934967041016 seconds
embed_image took 1.264251708984375 seconds
search_similar_in_weaviate took 0.08868932723999023 seconds
```

### 4. 사이즈 3000*3000 업로드(약 500개 업로드), 검색
```
Issue9/locustResult/upConv3000.html
Issue9/locustResult/searchSim_Conv3000.html

다시 test
Issue9/locustResult/upConvN3000.html
Issue9/locustResult/searchSim_ConvN3000.html
```

```
read_image took 3.0994415283203125e-06 seconds
save_image took 0.001285552978515625 seconds
save_to_db took 0.007987737655639648 seconds
send_to_kafka took 0.00030493736267089844 seconds

read_image took 2.6226043701171875e-06 seconds
save_image took 0.004534482955932617 seconds
save_to_db took 0.006303310394287109 seconds
send_to_kafka took 0.00025200843811035156 seconds


다시 test
read_image took 3.5762786865234375e-06 seconds
save_image took 0.002202272415161133 seconds
save_to_db took 0.010223627090454102 seconds
send_to_kafka took 0.0021712779998779297 seconds

read_image took 2.6226043701171875e-06 seconds
save_image took 0.0005648136138916016 seconds
save_to_db took 0.005040884017944336 seconds
send_to_kafka took 0.0002548694610595703 seconds
```

```
preprocess_resize took 0.08348202705383301 seconds
preprocess_center_crop took 0.00030803680419921875 seconds
preprocess_to_tensor took 0.0013682842254638672 seconds
preprocess_normalize took 0.00036525726318359375 seconds
embed_image took 0.5384984016418457 seconds
search_similar_in_weaviate took 0.08428955078125 seconds

preprocess_resize took 0.08305978775024414 seconds
preprocess_center_crop took 0.0002760887145996094 seconds
preprocess_to_tensor took 0.004247188568115234 seconds
preprocess_normalize took 0.0004048347473144531 seconds
embed_image took 0.43246984481811523 seconds
search_similar_in_weaviate took 0.11831378936767578 seconds


다시 test
preprocess_resize took 0.09157037734985352 seconds
preprocess_center_crop took 0.0003120899200439453 seconds
preprocess_to_tensor took 0.002637147903442383 seconds
preprocess_normalize took 0.0006620883941650391 seconds
embed_image took 0.7266738414764404 seconds
search_similar_in_weaviate took 0.09537148475646973 seconds

preprocess_resize took 0.10226297378540039 seconds
preprocess_center_crop took 0.0006504058837890625 seconds
preprocess_to_tensor took 0.0015192031860351562 seconds
preprocess_normalize took 0.0005075931549072266 seconds
embed_image took 0.8967406749725342 seconds
search_similar_in_weaviate took 0.0889742374420166 seconds
```

** embed_image에서 조금 오래 걸림 **
ConvNeXt -> ResNet18 or ResNet 50 사용해보기


## ResNet18
### 1. 사이즈 100*100 업로드, 검색
```
Issue9/locustResult/upRes18_100.html
Issue9/locustResult/searchSim_Res18_100.html
```

```
read_image took 3.0994415283203125e-06 seconds
save_image took 0.0002377033233642578 seconds
save_to_db took 0.007140398025512695 seconds
send_to_kafka took 0.0003986358642578125 seconds

read_image took 3.337860107421875e-06 seconds
save_image took 0.0002377033233642578 seconds
save_to_db took 0.022011995315551758 seconds
send_to_kafka took 0.00046515464782714844 seconds

read_image took 2.384185791015625e-06 seconds
save_image took 0.00018548965454101562 seconds
save_to_db took 0.016785144805908203 seconds
send_to_kafka took 0.00047016143798828125 seconds
```

```
preprocess_resize took 0.0010161399841308594 seconds
preprocess_center_crop took 0.00012803077697753906 seconds
preprocess_to_tensor took 0.0043108463287353516 seconds
preprocess_normalize took 0.0012311935424804688 seconds
embed_image took 0.3343679904937744 seconds
search_similar_in_weaviate took 0.07427668571472168 seconds

preprocess_resize took 0.0010221004486083984 seconds
preprocess_center_crop took 0.00012087821960449219 seconds
preprocess_to_tensor took 0.001967191696166992 seconds
preprocess_normalize took 0.0003337860107421875 seconds
embed_image took 0.1406116485595703 seconds
```



### 2. 사이즈 1000*1000 업로드, 검색
```
Issue9/locustResult/upRes18_1000.html
Issue9/locustResult/searchSim_Res18_1000.html
```

```
read_image took 2.6226043701171875e-06 seconds
save_image took 0.0023658275604248047 seconds
save_to_db took 0.014706611633300781 seconds
send_to_kafka took 0.0011718273162841797 seconds

read_image took 3.0994415283203125e-06 seconds
save_image took 0.0014007091522216797 seconds
save_to_db took 0.02882695198059082 seconds
send_to_kafka took 0.00038695335388183594 seconds
```

```
preprocess_resize took 0.017490625381469727 seconds
preprocess_center_crop took 0.00028705596923828125 seconds
preprocess_to_tensor took 0.014545679092407227 seconds
preprocess_normalize took 0.024509191513061523 seconds
embed_image took 0.1705620288848877 seconds
search_similar_in_weaviate took 0.09185504913330078 seconds

preprocess_resize took 0.01097416877746582 seconds
preprocess_center_crop took 0.00041294097900390625 seconds
preprocess_to_tensor took 0.0035567283630371094 seconds
preprocess_normalize took 0.0003821849822998047 seconds
embed_image took 0.08251667022705078 seconds
search_similar_in_weaviate took 0.07481813430786133 seconds
```


### 3. 사이즈 3000*3000 업로드, 검색
```
Issue9/locustResult/upRes18_3000.html
Issue9/locustResult/searchSim_Res18_3000.html
```


```
read_image took 4.5299530029296875e-06 seconds
save_image took 0.00047588348388671875 seconds
save_to_db took 0.007956743240356445 seconds
send_to_kafka took 0.0004107952117919922 seconds

read_image took 5.9604644775390625e-06 seconds
save_image took 0.0022306442260742188 seconds
save_to_db took 0.0076487064361572266 seconds
send_to_kafka took 0.0004608631134033203 seconds
```

```
preprocess_resize took 0.11497974395751953 seconds
preprocess_center_crop took 0.0005393028259277344 seconds
preprocess_to_tensor took 0.01479196548461914 seconds
preprocess_normalize took 0.008984804153442383 seconds
embed_image took 0.5031085014343262 seconds
search_similar_in_weaviate took 0.07644081115722656 seconds

preprocess_resize took 0.08750796318054199 seconds
preprocess_center_crop took 0.0002689361572265625 seconds
preprocess_to_tensor took 0.0015883445739746094 seconds
preprocess_normalize took 0.00036454200744628906 seconds
embed_image took 0.13254237174987793 seconds
search_similar_in_weaviate took 0.06756472587585449 seconds
```

## ResNet50
### 1. 사이즈 100*100 업로드, 검색
```
Issue9/locustResult/upRes50_100.html
Issue9/locustResult/searchSim_Res50_100.html
```

```
read_image took 2.86102294921875e-06 seconds
save_image took 0.00046825408935546875 seconds
save_to_db took 0.008225679397583008 seconds
send_to_kafka took 0.00028228759765625 seconds

read_image took 4.76837158203125e-06 seconds
save_image took 0.00019931793212890625 seconds
save_to_db took 0.049735307693481445 seconds
send_to_kafka took 0.00033164024353027344 seconds
```

```
preprocess_resize took 0.0009999275207519531 seconds
preprocess_center_crop took 0.00012159347534179688 seconds
preprocess_to_tensor took 0.022625207901000977 seconds
preprocess_normalize took 0.0004284381866455078 seconds
embed_image took 0.3402290344238281 seconds
search_similar_in_weaviate took 0.0757606029510498 seconds

preprocess_resize took 0.0018596649169921875 seconds
preprocess_center_crop took 0.0002803802490234375 seconds
preprocess_to_tensor took 0.003114461898803711 seconds
preprocess_normalize took 0.00038909912109375 seconds
embed_image took 0.17344069480895996 seconds
search_similar_in_weaviate took 0.08798503875732422 seconds
```



### 2. 사이즈 1000*1000 업로드, 검색
```
Issue9/locustResult/upRes50_1000.html
Issue9/locustResult/searchSim_Res50_1000.html
```

```
read_image took 2.6226043701171875e-06 seconds
save_image took 0.00019359588623046875 seconds
save_to_db took 0.012894153594970703 seconds
send_to_kafka took 0.0003464221954345703 seconds

read_image took 5.626678466796875e-05 seconds
save_image took 0.0003566741943359375 seconds
save_to_db took 0.010935783386230469 seconds
send_to_kafka took 0.006768465042114258 seconds
```

```
preprocess_resize took 0.009776115417480469 seconds
preprocess_center_crop took 0.0002562999725341797 seconds
preprocess_to_tensor took 0.0011088848114013672 seconds
preprocess_normalize took 0.0003662109375 seconds
embed_image took 0.30067014694213867 seconds
search_similar_in_weaviate took 0.09008550643920898 seconds

preprocess_resize took 0.010282754898071289 seconds
preprocess_center_crop took 0.0003571510314941406 seconds
preprocess_to_tensor took 0.0009300708770751953 seconds
preprocess_normalize took 0.0003254413604736328 seconds
embed_image took 0.39661288261413574 seconds
search_similar_in_weaviate took 0.07583951950073242 seconds
```


### 3. 사이즈 3000*3000 업로드, 검색
```
Issue9/locustResult/upRes50_3000.html
Issue9/locustResult/searchSim_Res50_3000.html
```


```
read_image took 2.86102294921875e-06 seconds
save_image took 0.0005614757537841797 seconds
save_to_db took 0.00751042366027832 seconds
send_to_kafka took 0.0003829002380371094 seconds

read_image took 2.86102294921875e-06 seconds
save_image took 0.0011138916015625 seconds
save_to_db took 0.011119365692138672 seconds
send_to_kafka took 0.0012621879577636719 seconds
```

```
preprocess_resize took 0.08030509948730469 seconds
preprocess_center_crop took 0.0002589225769042969 seconds
preprocess_to_tensor took 0.005683422088623047 seconds
preprocess_normalize took 0.0004096031188964844 seconds
embed_image took 0.2500481605529785 seconds
search_similar_in_weaviate took 0.07847809791564941 seconds

preprocess_resize took 0.09460592269897461 seconds
preprocess_center_crop took 0.0002658367156982422 seconds
preprocess_to_tensor took 0.0009446144104003906 seconds
preprocess_normalize took 0.0003218650817871094 seconds
embed_image took 0.26516222953796387 seconds
search_similar_in_weaviate took 0.07011723518371582 seconds
```