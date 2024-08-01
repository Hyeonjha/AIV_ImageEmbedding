import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import timm
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import os
import ssl
from typing import Optional
import torch.nn.functional as F
from PIL import Image
import argparse

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

# Focal Loss 관련 코드 추가
def label_to_one_hot_label(labels: torch.Tensor, num_classes: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, eps: float = 1e-6, ignore_index=255) -> torch.Tensor:
    shape = labels.shape
    one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]
    return ret

def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")
    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')
    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
    
    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
        
    input_soft = F.softmax(input, dim=1) + eps
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device, dtype=input.dtype, ignore_index=ignore_index)
    weight = torch.pow(1.0 - input_soft, gamma)
    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma = 2.0, reduction = 'mean', eps = 1e-8, ignore_index=30):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)

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
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean').to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        train_losses = []
        test_accuracies = []
        
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
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

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
