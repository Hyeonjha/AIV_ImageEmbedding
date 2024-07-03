import weaviate
import os

WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')

client = weaviate.Client(WEAVIATE_URL)

# 클래스 이름 정의 (예: 'ImageEmbedding')
class_name = 'ImageEmbedding'

# Weaviate에서 모든 객체 삭제
def delete_all_objects():
    try:
        schema = client.schema.get()
        if class_name not in [cls['class'] for cls in schema['classes']]:
            print(f"No schema found for class '{class_name}'. Please ensure the schema is imported first.")
            return

        result = client.query.aggregate(class_name).with_meta_count().do()
        total_count = result["data"]["Aggregate"][class_name][0]["meta"]["count"]
        print(f"Total objects to delete: {total_count}")

        batch_size = 100
        offset = 0

        while offset < total_count:
            objects = client.query.get(class_name).with_limit(batch_size).with_offset(offset).do()
            for obj in objects["data"]["Get"][class_name]:
                client.data_object.delete(obj["id"])
            offset += batch_size

        print("All objects deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    delete_all_objects()
