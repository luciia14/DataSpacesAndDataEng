from pathlib import Path
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.vision.image_dataset import EuroSATDataset

# Configuración
TRAIN_DIR = Path("data/processed/images/train")
TEST_DIR = Path("data/processed/images/test")
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001
MODEL_PATH = Path("models/resnet18_finetuned.pt")
CLASS_NAMES_PATH = Path("models/resnet18_finetuned_classes.txt")
REPORT_PATH = Path("reports/fine_tuning_report.txt")
CONFUSION_MATRIX_PATH = Path("reports/fine_tuning_confusion_matrix.png")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

    return train_loader, test_loader, train_dataset.class_names


def create_transfer_model(num_classes):
    # Cargar ResNet18 con pesos pre-entrenados
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )
    
    # 1. Congelar todos los parámetros inicialmente
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    # 2. Descongelar la layer4 (Fine-tuning parcial)
    # Esto permite que las capas convolucionales finales aprendan rasgos específicos de EuroSAT
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
        
    # 3. Reemplazar la cabeza del clasificador
    input_features = model.fc.in_features
    model.fc = nn.Linear(
        input_features,
        num_classes
    )
    
    return model

def print_trainable_parameters(model):
    print("=== Trainable Parameters ===")
    trainable_count = 0
    total_count = 0
    for name, parameter in model.named_parameters():
        total_count += parameter.numel()
        if parameter.requires_grad:
            trainable_count += parameter.numel()
            print(name)
    print(f"Trainable parameters: {trainable_count}")
    print(f"Total parameters: {total_count}")

def train_model(model, train_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad, 
        model.parameters()
    ),
    lr=LEARNING_RATE
)
    
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")
        
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    return training_time

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(
                outputs,
                dim=1
            )
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )
    
    print("=== Transfer Learning Evaluation ===")
    print(f"Test samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )
    
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )
    
    # Asegura que la carpeta reports/ existe
    CONFUSION_MATRIX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    
    display.plot()
    plt.title("Transfer Learning Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")
    
    return accuracy, all_labels, all_predictions
def save_model(model, class_names):
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    torch.save(
        model.state_dict(),
        MODEL_PATH
    )
    with open(CLASS_NAMES_PATH, "w") as f:
        for class_name in class_names:
            f.write(class_name + "\n")
    print("=== Saving Transfer Model ===")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved classes: {CLASS_NAMES_PATH}")

def save_report(accuracy, training_time, class_names):
    REPORT_PATH.parent.mkdir(
        parents=True, 
        exist_ok=True
    )
    with open(REPORT_PATH, "w") as f:
        f.write("TRANSFER LEARNING REPORT\n")
        f.write("========================\n\n")
        f.write("Model: ResNet18 pretrained on ImageNet\n")
        f.write("Training mode: fine-tuned last ResNet block\n")
        f.write("Trainable layers: layer4 and final classifier\n\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write(
            "The model reused pretrained visual features and trained only "
            "a new classifier head for the EuroSAT subset.\n"
        )
    print(f"Saved report: {REPORT_PATH}")

def main():
    # 1. Preparar datos y obtener nombres de clases
    train_loader, test_loader, class_names = create_dataloaders()
    
    # 2. Configurar el dispositivo (CPU o GPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # 3. Crear el modelo de Transfer Learning
    model = create_transfer_model(
        num_classes=len(class_names)
    )
    
    # 4. Verificar qué parámetros se van a entrenar (Task 8 & 9)
    print_trainable_parameters(model)
    
    # 5. Mover el modelo al dispositivo configurado
    model = model.to(device)
    
    # 6. Entrenar la "cabeza" del clasificador (Task 10)
    training_time = train_model(
        model, 
        train_loader, 
        device
    )
    
    # 7. Evaluar el rendimiento y generar matriz de confusión
    accuracy, all_labels, all_predictions = evaluate_model(
        model,
        test_loader,
        device,
        class_names
    )
    
    # 8. Guardar los pesos del modelo y las etiquetas
    save_model(
        model, 
        class_names
    )
    
    # 9. Generar el informe final en formato .txt
    save_report(
        accuracy,
        training_time,
        class_names
    )

if __name__ == "__main__":
    main()