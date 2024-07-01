import torch
import torchvision.transforms as transforms
import timm
from PIL import Image

class Img2Vec:
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
