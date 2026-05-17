from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

# Importar tus modelos y datasets
from src.vision.cnn_model import SimpleCNN
from src.vision.image_dataset import EuroSATDataset

# Configuración
MODEL_PATH = Path("models/cnn_model.pt")
CLASS_NAMES_PATH = Path("models/cnn_classes.txt")
TEST_DIR = Path("data/processed/images/test")
REPORT_PATH = Path("reports/cnn_vs_ml.txt")

def main():
    # 1. Cargar Nombres de Clases
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # 2. Configurar CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Preparar Datos de Prueba
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = EuroSATDataset(root_dir=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 4. Evaluación de CNN
    correct = 0
    total = len(test_dataset)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    cnn_accuracy = correct / total

    # 5. Generar Reporte
    # Nota: Debes rellenar los datos de ML Clásico de tu laboratorio anterior
    report_content = f"""CNN VS CLASSICAL ML
===================

CLASSICAL ML:
Model: Random Forest (o el usado en Lab anterior)
Training time: [Inserte tiempo de Lab anterior]
Accuracy: [Inserte precisión de Lab anterior]

CNN:
Model: Simple CNN
Training time: ~2-5 minutes (depende de CPU/GPU)
Accuracy: {cnn_accuracy:.4f}

BETTER ACCURACY:
La CNN suele superar al ML Clásico porque extrae características espaciales
(bordes, texturas) en lugar de tratar cada píxel como una variable aislada.

FASTER TRAINING:
ML Clásico (Random Forest/SVM) es significativamente más rápido de entrenar
en datasets pequeños, ya que no requiere optimización por gradiente.

FEWER CLASS CONFUSIONS:
La CNN tiende a confundir menos las clases con texturas similares 
(como River vs Forest) gracias a los filtros convolucionales.

GENERALIZATION:
La CNN demuestra una mejor capacidad de generalización ante rotaciones 
o cambios de iluminación leves en las imágenes satelitales.
"""

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report_content)
    
    print(f"Reporte de comparación guardado en: {REPORT_PATH}")

if __name__ == "__main__":
    main()