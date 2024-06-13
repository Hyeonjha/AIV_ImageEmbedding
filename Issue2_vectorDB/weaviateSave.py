import weaviate
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import os

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

def save_image_embedding(folder_path):
    img2vec = Img2Vec()
    client = weaviate.Client("http://localhost:8080")

    try:
        # 클래스가 이미 존재하는지 확인
        existing_schema = client.schema.get()
        class_exists = any(cls["class"] == "Image" for cls in existing_schema["classes"])
    except KeyError:
        class_exists = False

    if not class_exists:
        # Weaviate 스키마 생성
        client.schema.create_class({
            "class": "Image",
            "description": "An image object",
            "properties": [
                {
                    "name": "image_path",
                    "description": "The path to the image file",
                    "dataType": ["text"]
                },
                {
                    "name": "label",
                    "description": "The label of the image",
                    "dataType": ["text"]
                },
                {
                    "name": "embedding",
                    "description": "The vector embedding of the image",
                    "dataType": ["number[]"]  # 리스트 형태의 숫자 타입으로 변경
                }
            ]
        })

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                img = Image.open(img_path).convert('RGB')
                embedding = img2vec.get_vec(img)
                client.data_object.create(
                    class_name="Image",
                    data_object={
                        "image_path": img_path,
                        "label": label,
                        "embedding": embedding
                    }
                )

if __name__ == "__main__":
    folder_path = 'data-gatter/train_L'
    save_image_embedding(folder_path)
