### producer.py
### json 형태의 데이터 1000개 생성하여 topic 1이라는 topic으로 메세지 전송 코드
### 각 메시지는 JSON 형식으로 직렬화되어 gzip으로 압축됨

from kafka import KafkaProducer
from json import dumps  # 데이터를 JSON 형식으로 직렬화하기 위한 함수
import time
 
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'], # 전달하고자 하는 카프카 브로커의 주소 리스트 (여기선 로컬호스트의 기본 포트 사용)
                                          # 되도록이면 2개 이상의 ip와 port 설정하도록 권장
                                          # 둘 중 하나 비정상일 경우, 다른 하나에 연결되어 사용하게끔!
    acks=0, # 메시지 전송 완료에 대한 체크
    compression_type='gzip', # 메시지 전달할 때 gzip 형식으로 압축(None, gzip, snappy, lz4 등)
    value_serializer=lambda x:dumps(x).encode('utf-8') # 메시지의 값 JSON 형식으로 '직렬화'하고 UTF-8 인코딩
)
 
start = time.time()
 
for i in range(1000):
    data = {'str' : 'result'+str(i)}  # 전송할 데이터 생성. (문자열 "result"와 루프 인덱스를 합친 문자열을 포함하는 딕셔너리.)
    producer.send('topic1', value=data)  # 'topic1' 토픽에 데이터(value)를 전송
                                         # key 포함하여 전송 : send('topic1', '1', value=data)
    #producer.flush() # 버퍼에 있는 모든 메시지를 브로커로 강제 전송. 모든 전송이 완료될 때까지 블록.
                     # 여기서 flush()를 매번 호출하면 성능이 저하될 수 있음
producer.flush()

print('[Done]:', time.time() - start)