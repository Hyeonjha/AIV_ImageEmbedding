import weaviate

def get_saved_embeddings():
    client = weaviate.Client("http://localhost:8080")

    # 모든 이미지 객체 가져오기
    result = client.query.get(
        "Image",
        ["image_path", "label", "embedding"]
    ).do()

    for obj in result["data"]["Get"]["Image"]:
        print(f"Image Path: {obj['image_path']}")
        print(f"Label: {obj['label']}")
        print(f"Embedding: {obj['embedding'][:5]}...")  # 임베딩의 첫 5개 요소만 출력

if __name__ == "__main__":
    get_saved_embeddings()
