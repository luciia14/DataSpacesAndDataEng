from pathlib import Path
from PIL import Image
import random
import shutil

# --- Task 2.2: Configuración ---
RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed/images")
SELECTED_CLASSES = {
    "Forest": "forest",
    "River": "river",
    "Residential": "residential"
}
IMAGES_PER_CLASS = 60
TRAIN_RATIO = 0.8
TARGET_SIZE = (64, 64)
RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# --- Task 2.3: Buscar la carpeta de EuroSAT ---
def find_eurosat_folder(raw_root):
    for path in raw_root.rglob("*"):
        if path.is_dir():
            class_names = [
                child.name for child in path.iterdir()
                if child.is_dir()
            ]
            # Verifica si las clases seleccionadas están en esta carpeta
            if all(class_name in class_names for class_name in SELECTED_CLASSES.keys()):
                return path
    
    print("Error: Could not find EuroSAT class folders.")
    print("Make sure you ran src/data/download_eurosat.py first.")
    raise SystemExit(1)

# --- Task 2.4: Preparar carpetas de salida ---
def prepare_output_directories():
    if PROCESSED_ROOT.exists():
        shutil.rmtree(PROCESSED_ROOT)
    
    for split_name in ["train", "test"]:
        for output_class in SELECTED_CLASSES.values():
            output_dir = PROCESSED_ROOT / split_name / output_class
            output_dir.mkdir(parents=True, exist_ok=True)

# --- Task 2.5: Procesar una imagen ---
def process_single_image(input_path, output_path):
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        image = image.resize(TARGET_SIZE)
        image.save(output_path)

# --- Task 2.6: Preparar el dataset completo ---
def prepare_dataset():
    random.seed(RANDOM_SEED)
    eurosat_folder = find_eurosat_folder(RAW_ROOT)
    prepare_output_directories()
    
    print("=== Preparing Image Dataset ===")
    print(f"EuroSAT folder: {eurosat_folder}")
    print(f"Processed dataset: {PROCESSED_ROOT}")
    
    for original_class, output_class in SELECTED_CLASSES.items():
        class_dir = eurosat_folder / original_class
        image_files = [
            path for path in class_dir.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        random.shuffle(image_files)
        selected_files = image_files[:IMAGES_PER_CLASS]
        
        split_index = int(TRAIN_RATIO * len(selected_files))
        train_files = selected_files[:split_index]
        test_files = selected_files[split_index:]
        
        print(f"\nClass: {original_class}")
        print(f"Selected images: {len(selected_files)}")
        print(f"Training images: {len(train_files)}")
        print(f"Testing images: {len(test_files)}")
        
        # Procesamiento y guardado
        for split_name, files in [("train", train_files), ("test", test_files)]:
            for index, input_path in enumerate(files):
                output_filename = f"{output_class}_{index:04d}.jpg"
                output_path = (
                    PROCESSED_ROOT 
                    / split_name 
                    / output_class 
                    / output_filename
                )
                process_single_image(input_path, output_path)

# --- Task 2.7: Punto de entrada ---
def main():
    prepare_dataset()

if __name__ == "__main__":
    main()