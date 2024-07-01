import weaviate

# Weaviate 클라이언트 설정
client = weaviate.Client(url="http://localhost:8080")

def check_embeddings():
    try:
        result = client.query.get("ImageEmbedding", ["image_id", "_additional {vector}"]).do()
        
        if 'data' not in result:
            print("Error: 'data' key not found in query result")
            print(result)
            return

        embeddings = result['data']['Get']['ImageEmbedding']
        
        for embedding in embeddings:
            image_id = embedding['image_id']
            vector = embedding['_additional']['vector']
            print(f"Image ID: {image_id}")
            print(f"Embedding Vector: {vector}")
            print("="*50)
            
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    check_embeddings()
