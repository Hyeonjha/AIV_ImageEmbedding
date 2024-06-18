import weaviate

# Weaviate 클라이언트 설정
client = weaviate.Client(url="http://localhost:8080")

# 데이터베이스 초기화 함수
def initialize_database(client):
    # 모든 클래스 삭제
    try:
        schema = client.schema.get()
        for class_obj in schema['classes']:
            client.schema.delete_class(class_obj['class'])
        print("Database initialized.")
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    initialize_database(client)
