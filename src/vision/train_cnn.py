from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.vision.image_dataset import EuroSATDataset
from src.vision.cnn_model import SimpleCNN

TRAIN_DIR = Path("data/processed/images/train")
TEST_DIR = Path("data/processed/images/test")
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = Path("models/cnn_model.pt")
CLASS_NAMES_PATH = Path("models/cnn_classes.txt")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataloaders():
    # Training transform with Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    
    # Testing transform (keep original, no augmentation for evaluation)
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = EuroSATDataset(root_dir=TRAIN_DIR, transform=train_transform)
    test_dataset = EuroSATDataset(root_dir=TEST_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, train_dataset.class_names

def train_model(model, train_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    print("\nStarting Training with Data Augmentation...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print("\n=== Evaluation ===")
    print(f"Test samples: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def save_model(model, class_names):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        for class_name in class_names:
            f.write(class_name + "\n")
    print("\n=== Saving Model & Classes ===")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved classes: {CLASS_NAMES_PATH}")

def main():
    train_loader, test_loader, class_names = create_dataloaders()
    device = get_device()
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=len(class_names)).to(device)

    # We train the model to apply the new augmented patterns
    train_model(model, train_loader, device)
    
    evaluate_model(model, test_loader, device)
    save_model(model, class_names)

if __name__ == "__main__":
    main()