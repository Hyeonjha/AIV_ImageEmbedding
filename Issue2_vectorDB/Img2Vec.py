### Programmer : Hyeon Ji Ha
### Date : Jun 14 2024
### Purpose : 이미지로부터 특징 벡터(임베딩)를 추출하는 기능
###           

import torch
import torchvision.transforms as transforms
import timm

# 이미지 파일을 로드하고 모델을 통해 벡터로 변환하는 기능을 제공
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
        return embedding_np