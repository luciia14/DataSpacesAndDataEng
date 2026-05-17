from pathlib import Path
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Configuración
MODEL_PATH = Path("models/resnet18_transfer.pt")
CLASS_NAMES_PATH = Path("models/resnet18_classes.txt")

def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        print(f"Error: class file not found: {CLASS_NAMES_PATH}")
        print("Train the transfer model first.")
        raise SystemExit(1)
    
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]
    return class_names

def load_model(class_names):
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Train the transfer model first.")
        raise SystemExit(1)
    
    # Creamos la estructura de ResNet18 sin pesos pre-entrenados
    model = models.resnet18(weights=None)
    
    # Debemos replicar exactamente la arquitectura del Task 9
    input_features = model.fc.in_features
    model.fc = nn.Linear(
        input_features,
        len(class_names)
    )
    
    # Cargamos los pesos guardados en el entrenamiento
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    
    # Ponemos el modelo en modo evaluación
    model.eval()
    return model

def predict_image(model, class_names, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: image not found: {path}")
        return
    
    # Transformaciones necesarias para que la imagen sea compatible con ResNet18
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])    
    with Image.open(path) as image:
        image = image.convert("RGB")
        image_for_plot = image.copy()
        
        # Aplicamos transformaciones y añadimos la dimensión del batch (unsqueeze)
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(
            outputs,
            dim=1
        )
        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )
        
    predicted_class = class_names[
        predicted_index.item()
    ]
    
    print("=== Transfer Learning Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")
    
    # Visualización
    plt.imshow(image_for_plot)
    plt.title(
        f"Transfer model prediction: {predicted_class}\n"
        f"Confidence: {confidence.item():.4f}"
    )
    plt.axis("off")
    plt.show()

def main():
    # 1. Cargamos etiquetas y modelo
    class_names = load_class_names()
    model = load_model(class_names)
    
    # 2. Ruta de la imagen para predecir
    image_path = "data/inference_samples/noise.jpg"
    
    # 3. Ejecutamos la predicción
    predict_image(
        model,
        class_names,
        image_path
    )

if __name__ == "__main__":
    main()