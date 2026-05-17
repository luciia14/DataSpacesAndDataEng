from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.vision.image_dataset import EuroSATDataset
from src.vision.cnn_model import SimpleCNN

# Configuration
TRAIN_DIR = Path("data/processed/images/train")
TEST_DIR = Path("data/processed/images/test")
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001
MODEL_PATH = Path("models/cnn_model.pt")
CLASS_NAMES_PATH = Path("models/cnn_classes.txt")

def get_device():
    """Selects CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_dataloaders():
    """Initializes the dataset and creates PyTorch DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = EuroSATDataset(
        root_dir=TRAIN_DIR, 
        transform=transform
    )
    test_dataset = EuroSATDataset(
        root_dir=TEST_DIR, 
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    print("=== DataLoader Inspection ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.class_names}")
    
    return train_loader, test_loader, train_dataset.class_names

def train_model(model, train_loader, device):
    """Executes the training loop over the specified number of epochs."""
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")

def main():
    # 1. Prepare Data
    train_loader, test_loader, class_names = create_dataloaders()
    
    # 2. Setup Hardware
    device = get_device()
    print(f"Using device: {device}")

    # 3. Initialize Model
    model = SimpleCNN(
        num_classes=len(class_names)
    ).to(device)

    # 4. Train
    print("\nStarting Training...")
    train_model(
        model,
        train_loader,
        device
    )
    print("Training Complete.")

if __name__ == "__main__":
    main()