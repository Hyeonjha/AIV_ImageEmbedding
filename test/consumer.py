### consumer.py
### json 형태의 데이터 1000개 생성하여 topic 1이라는 topic으로 메세지 전송 코드
### https://dev-records.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-Kafka-%EA%B0%84%EB%8B%A8%ED%95%9C-%EC%98%88%EC%A0%9C

from kafka import KafkaConsumer
from json import loads  # JSON 형식의 데이터를 파이썬 객체로 역직렬화하기 위한 함수

consumer = KafkaConsumer(
    'topic1', # 소비할 Kafka 토픽명
    bootstrap_servers=['localhost:9092'], # 카프카 브로커 설정. 브로커 중 한개에 이슈가 생기면 다른 브로커가 붙을 수 있도록 여러개 지정 추천
    group_id='test-group', # 컨슈머 그룹 식별자 (동일한 그룹에 속한 컨슈머는 메시지를 분산하여 처리)
    auto_offset_reset='earliest', # 오프셋 위치(earliest:가장 처음부터 메세지 읽기 시작, latest: 가장 최근부터)
    enable_auto_commit=True, # 오프셋 자동으로 커밋 (메세지 읽을 때마다 소비된 위치 자동 저장)
    value_deserializer=lambda x: loads(x.decode('utf-8')), # 메시지의 값 역직렬화
    consumer_timeout_ms=1000 # 데이터를 기다리는 최대 시간. 이 시간 동안 데이터 없으면 'StopIteration' 예외 발생시킴
)

print('[Start] get consumer')

for message in consumer:
    print(f'Topic : {message.topic}, Partition : {message.partition}, Offset : {message.offset}, Key : {message.key}, value : {message.value}')


print('[End] get consumer')
