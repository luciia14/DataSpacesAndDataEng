from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Configuración de Rutas
MODEL_PATH = Path("models/resnet18_transfer.pt")
CLASS_NAMES_PATH = Path("models/resnet18_classes.txt")

# Task 5: Load Trained Model
def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        class_names = [
            line.strip()
            for line in f
            if line.strip()
        ]
    return class_names

def load_model(class_names):
    model = models.resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(class_names))
    
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Task 6: Load Image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image)
    return image, tensor.unsqueeze(0)

# Task 7: Generate Prediction
def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        predicted_class = class_names[predicted.item()]
        
    print(f"Prediction: {predicted_class} | Confidence: {confidence.item():.4f}")
    return predicted_class, confidence.item()

# Task 8: Generate Grad-CAM Heatmap
def create_heatmap(model, image_tensor):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam[0]

# Task 9: Visualize and Save Transform Pair
def visualize_transformation(original_img, transformed_tensor, heatmap, transform_name, output_path):
    # Convertir el tensor transformado de vuelta a imagen PIL para visualizar el fondo original real
    # Deshacemos el batch dimension y pasamos a numpy HWC
    t_np = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    t_np = np.clip(t_np, 0, 1) # Asegurar rango válido [0, 1]
    
    visualization = show_cam_on_image(t_np, heatmap, use_rgb=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(t_np)
    plt.title(f"Transformed: {transform_name}")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM ({transform_name})")
    plt.axis("off")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization: {output_path}")

# Task 10 / Independent Task 2: Main Function
def main():
    class_names = load_class_names()
    model = load_model(class_names)
    
    # Imagen de referencia (usamos river_0000 como base, puedes cambiarla si quieres)
    base_image_path = Path("data/processed/images/test/river/river_0000.jpg")
    
    if not base_image_path.exists():
        print(f"Error: Base image not found at {base_image_path}")
        return
        
    print(f"--- INDEPENDENT TASK 2: IMAGE TRANSFORMATION SENSITIVITY ---")
    print(f"Base Image: {base_image_path}\n")
    
    # 1. Carga original limpia
    orig_img, orig_tensor = load_image(base_image_path)
    
    # Definimos las diferentes transformaciones aplicadas directamente al tensor base [1, 3, 224, 224]
    # Clonamos para no pisar memoria entre experimentos
    transformations = {}
    
    # Original
    transformations["Original"] = orig_tensor.clone()
    
    # Horizontal Flip
    transformations["Horizontal_Flip"] = TF.hflip(orig_tensor.clone())
    
    # Rotation 90 degrees
    transformations["Rotation_90"] = TF.rotate(orig_tensor.clone(), angle=90)
    
    # Gaussian Blur (kernel_size=11, sigma=3.0)
    transformations["Gaussian_Blur"] = TF.gaussian_blur(orig_tensor.clone(), kernel_size=[11, 11], sigma=[3.0, 3.0])
    
    # Brightness Adjustment (Factor 1.7 -> Más brillante)
    transformations["High_Brightness"] = TF.adjust_brightness(orig_tensor.clone(), brightness_factor=1.7)
    
    # Additional Random Noise (Gaussian Noise)
    noise_tensor = orig_tensor.clone()
    noise = torch.randn_like(noise_tensor) * 0.15 # Magnitud del ruido aleatorio
    transformations["Gaussian_Noise"] = torch.clamp(noise_tensor + noise, 0.0, 1.0)

    # Bucle para ejecutar predicción y Grad-CAM sobre cada una de las mutaciones
    for t_name, t_tensor in transformations.items():
        print(f"Executing experiment for mutation: {t_name}")
        
        # Ejecutar inferencia
        pred_class, conf_score = predict(model, t_tensor, class_names)
        
        # Generar su mapa de calor correspondiente
        heatmap = create_heatmap(model, t_tensor)
        
        # Guardar la gráfica comparativa con sufijo dinámico
        output_path = Path(f"reports/gradcam_examples/transformation_{base_image_path.stem}_{t_name}.png")
        visualize_transformation(orig_img, t_tensor, heatmap, t_name, output_path)
        print("-" * 60)

if __name__ == "__main__":
    main()