import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np

class SignatureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = []

        for cls in sorted(os.listdir(root_dir)):
            person_id = cls.split('-')[0]
            signature_type = cls.split('-')[-1] if '-' in cls else 'genuine'

            cls_path = os.path.join(root_dir, cls)
            if os.path.isdir(cls_path):
                label = 1 if signature_type == 'f' else 0
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)
                    if cls not in self.classes:
                        self.classes.append(cls)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = plt.imread(img_path)

        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

class SignatureDecayModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SignatureDecayModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_accuracy = 0.0
    patience = 5
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            trigger_times = 0
            torch.save(model.state_dict(), 'model/signature_verification_model.pt')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    return model

def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SignatureDataset(root_dir='train', transform=transform)
    val_dataset = SignatureDataset(root_dir='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = SignatureDecayModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    trained_model = train_model(train_loader, val_loader, model, criterion, optimizer)

    print("Training completed. Best model saved as 'model/signature_verification_model.pt'")
    print("\nSignature Classes Used:")
    print(train_dataset.classes)


if __name__ == '__main__':
    main()
