import weaviate
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image

class Img2Vec():
    def __init__(self, model_name='convnext_base', cuda=False):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        self.model.eval()

    def get_vec(self, img):
        image = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image)
            embedding_np = embedding.cpu().numpy().flatten()
        return embedding_np.tolist()  # 리스트 형태로 변환

def find_similar_images(input_image_path, threshold=0.8, top_n=5):
    img = Image.open(input_image_path).convert('RGB')
    img2vec = Img2Vec()
    input_embedding = img2vec.get_vec(img)

    client = weaviate.Client("http://localhost:8080")

    # 유사도 검색 쿼리 실행
    result = client.query.get(
        "Image",
        ["image_path", "label", "_additional { distance }"]
    ).with_near_vector({
        "vector": input_embedding,
        "certainty": threshold
    }).with_limit(top_n).do()

    similar_images = [(obj["image_path"], obj["label"], obj["_additional"]["distance"]) for obj in result["data"]["Get"]["Image"]]
    return similar_images

if __name__ == "__main__":
    input_image_path = 'data-gatter/testcopy/bubble_381007.jpg'
    similar_images = find_similar_images(input_image_path, threshold=0.8, top_n=5)
    for image_path, label, distance in similar_images:
        print(f"Image Path: {image_path}, Label: {label}, Distance: {distance}")
