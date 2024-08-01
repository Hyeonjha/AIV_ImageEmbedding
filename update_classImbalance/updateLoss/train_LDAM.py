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
import argparse
from losses import LDAMLoss

# SSL 인증서 설정
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# Seed 설정
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class CustomDataset(Dataset):
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
                label = filename.split('_')[0]
                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)
                self.images.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]

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

def adjust_learning_rate(optimizer, epoch, lr):
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 180:
        lr = lr * 0.0001
    elif epoch > 160:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

    def train(self, train_loader, test_loader, epochs=10, lr=0.0001):
        cls_num_list = [len(np.where(np.array(train_loader.dataset.targets) == i)[0]) for i in range(len(train_loader.dataset.classes))]
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        train_losses = []
        test_accuracies = []
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            adjust_learning_rate(optimizer, epoch, lr)
            
            if epoch < 160:
                per_cls_weights = None
            else:
                beta = 0.9999
                effective_num = 1.0 - np.power(beta, cls_num_list)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
                criterion.weight = per_cls_weights

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
            train_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            # Calculate test accuracy
            test_accuracy = self.evaluate_accuracy(test_loader)
            test_accuracies.append(test_accuracy)
            print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {test_accuracy:.4f}")
        
        return train_losses, test_accuracies
    
    def evaluate_accuracy(self, test_loader):
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

        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        return accuracy
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier")
    parser.add_argument('--train_folder', type=str, required=True, help='Path to the training data folder')
    parser.add_argument('--test_folder', type=str, required=True, help='Path to the test data folder')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')  # 추가된 부분
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  # 추가된 부분
    args = parser.parse_args()

    set_seed(args.seed)  # Seed 설정

    train_loader, test_loader, class_names = load_data(args.train_folder, args.test_folder, batch_size=args.batch_size)

    model_dict = {
        'EfficientNetV2': 'efficientnet_v2_s',
        'EfficientNetB0': 'efficientnet_b0'
    }

    cuda = torch.cuda.is_available()

    for model_name, model in model_dict.items():
        print(f"Training and evaluating model {model_name}")
        classifier = Classifier(model, num_classes=len(class_names), pretrained=False, cuda=cuda)
        print(f"Starting training for model {model_name}...")
        train_losses, test_accuracies = classifier.train(train_loader, test_loader, epochs=args.epochs, lr=args.lr)
        print(f"Finished training for model {model_name}")

        # 평가
        print(f"Starting evaluation for model {model_name}...")
        cm, report = classifier.evaluate(test_loader, class_names)
        print(f"Finished evaluation for model {model_name}")
        print(f"\nModel: {model_name}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{report}")

        # 학습 손실 및 테스트 정확도 출력
        print(f"Train Losses: {train_losses}")
        print(f"Test Accuracies: {test_accuracies}")
