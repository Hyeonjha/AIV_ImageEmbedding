import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import timm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import ssl
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# SSL 인증서 설정
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

class CustomDataset(Dataset):
    # def __init__(self, folder, transform=None):
    #     self.images = []
    #     self.labels = []
    #     self.transform = transform
    #     valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    #     for filename in os.listdir(folder):
    #         img_path = os.path.join(folder, filename)
    #         if os.path.splitext(filename)[1].lower() in valid_image_extensions:
    #             img = Image.open(img_path).convert('RGB')
    #             label = filename.split('_')[0]  # Assuming the true label is encoded in the filename
    #             self.images.append(img)
    #             self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    # def __getitem__(self, idx):
    #     img = self.images[idx]
    #     label = self.labels[idx]
    #     if self.transform:
    #         img = self.transform(img)
    #     return img, label

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]

    def __init__(self, folder, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {}
        valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if os.path.splitext(filename)[1].lower() in valid_image_extensions:
                img = Image.open(img_path).convert('RGB')
                label = filename.split('_')[0]  # Assuming the true label is encoded in the filename
                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)
                self.images.append(img)
                self.labels.append(label)

def load_data(train_folder, test_folder, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(root=train_folder, transform=transform)
    test_dataset = CustomDataset(folder=test_folder, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.classes

class Classifier():
    def __init__(self, model_name, num_classes, pretrained=True, cuda=False):
        self.device = torch.device("cuda" if cuda else "cpu")
        if model_name == 'efficientnet_v2_s':
            self.model = models.efficientnet_v2_s(pretrained=pretrained).to(self.device)
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes).to(self.device)
        elif model_name.startswith('efficientnet'):
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes).to(self.device)
        else:
            raise ValueError(f"Model {model_name} not supported")
        self.model = self.model.to(self.device)

    def train(self, train_loader, epochs=10, lr=0.001):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
                if i % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    def evaluate(self, test_loader, class_names):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if i % 10 == 0:
                    print(f"Processed {i + 1}/{len(test_loader)} test batches")

        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=class_names)
        return cm, report

def visualize_confusion_matrix(cm, model_name, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

if __name__ == "__main__":
    train_folder = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/TRAIN_SET_R'
    test_folder = '/Users/hahyeonji/Documents/AIV_Intern/ImageImbedding/data-gatter/TEST_SET_A'
    model_dict = {
        'EfficientNetV2': 'efficientnet_v2_s',
        'EfficientNet B0': 'efficientnet_b0',
        'EfficientNet B7': 'efficientnet_b7'
    }

    train_loader, test_loader, class_names = load_data(train_folder, test_folder)

    cuda = torch.cuda.is_available()

    for model_name, model in model_dict.items():
        print(f"Training and evaluating model {model_name}")
        classifier = Classifier(model, num_classes=len(class_names), pretrained=False, cuda=cuda)
        print(f"Starting training for model {model_name}...")
        classifier.train(train_loader, epochs=1, lr=0.001)     #  epochs=10
        print(f"Finished training for model {model_name}")
        print(f"Starting evaluation for model {model_name}...")
        cm, report = classifier.evaluate(test_loader, class_names)
        print(f"Finished evaluation for model {model_name}")
        print(f"\nModel: {model_name}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{report}")
        visualize_confusion_matrix(cm, model_name, class_names)
