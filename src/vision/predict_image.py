from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from src.vision.feature_extractor import extract_features

MODELS_DIR = Path("models")

def predict_all_models(image_path):
    path = Path(image_path)
    if not path.exists(): return

    # Cargamos todos los modelos disponibles en la carpeta
    model_files = list(MODELS_DIR.glob("*.joblib"))
    predictions = []

    with Image.open(path) as image:
        features = extract_features(image)
        img_plot = image.copy()
        
        print(f"=== Predictions for: {path.name} ===")
        for m_file in model_files:
            model = joblib.load(m_file)
            pred = model.predict([features])[0]
            model_name = m_file.stem.replace('_', ' ').title()
            predictions.append(f"{model_name}: {pred}")
            print(f"{model_name}: {pred}")

    # Task 11.5: Visualización combinada
    title_text = " | ".join(predictions)
    plt.imshow(img_plot)
    plt.title(title_text, fontsize=9)
    plt.axis("off")
    plt.show()

def main():
    # Prueba con una imagen real de bosque
    img = "data/processed/images/test/forest/forest_0000.jpg"
    predict_all_models(img)

if __name__ == "__main__":
    main()