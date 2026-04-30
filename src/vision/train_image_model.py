import time
from pathlib import Path
from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from src.vision.feature_extractor import extract_features

DATASET_DIR = Path("data/processed/images")
MODELS_DIR = Path("models")

def load_image_split(split_dir):
    X, y = [], []
    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        for img_path in images:
            with Image.open(img_path) as img:
                X.append(extract_features(img))
                y.append(class_name)
    return np.array(X), np.array(y)

def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }
    
    results = []
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"{'Model':<20} | {'Accuracy':<10} | {'Train Time':<10}")
    print("-" * 45)

    for name, model in models.items():
        # Task 11.2: Measure Training Time
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Task 11.3: Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({"name": name, "accuracy": acc, "time": train_time})
        print(f"{name:<20} | {acc:<10.4f} | {train_time:<10.4f}s")
        
        # Guardamos cada modelo individualmente
        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    return results

def plot_results(results):
    # Task 11.4: Accuracy vs Training Time Plot
    names = [r['name'] for r in results]
    accs = [r['accuracy'] for r in results]
    times = [r['time'] for r in results]

    plt.figure(figsize=(10, 5))
    plt.scatter(times, accs, s=100, color='red')
    
    for i, name in enumerate(names):
        plt.annotate(name, (times[i], accs[i]), xytext=(5, 5), textcoords='offset points')

    plt.title("Model Comparison: Accuracy vs Training Time")
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def main():
    train_dir, test_dir = DATASET_DIR / "train", DATASET_DIR / "test"
    X_train, y_train = load_image_split(train_dir)
    X_test, y_test = load_image_split(test_dir)
    
    results = compare_models(X_train, X_test, y_train, y_test)
    plot_results(results)

if __name__ == "__main__":
    main()