### Delete Image Class - weaviate

import weaviate

client = weaviate.Client("http://localhost:8080")

# 클래스 삭제
client.schema.delete_class('Image')

print("Class 'Image' deleted successfully.")
