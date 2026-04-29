import csv
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

def run_model_training():
    # Rutas de archivos
    features_path = "data/processed/model_features.csv"
    labels_path = "data/processed/model_labels.csv"
    model_output_path = "results/decision_tree_model.joblib"
    evaluation_output_path = "results/model_evaluation.txt"
    report_output_path = "reports/model_training_summary.txt"
    
    # Asegurar que las carpetas existen
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # --- TASK 1: Loading Feature Dataset ---
    print("=== Machine Learning: Loading Feature Dataset ===")
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Error: Dataset files missing. Please run preprocessing first.")
        return

    features_raw = []
    column_names = []
    with open(features_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        column_names = reader.fieldnames
        for row in reader:
            features_raw.append(row)

    print(f"Input file: {features_path}")
    print(f"Records loaded: {len(features_raw)}")
    print(f"Columns: {column_names}")

    # --- TASK 2: Preparing Features (X) and Target (y) ---
    X = [[float(val) for val in row.values()] for row in features_raw]
    y = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            y.append(int(row['anomaly_flag']))

    print("\n=== Machine Learning: Preparing Features and Target ===")
    print(f"Number of samples in X: {len(X)}")
    print(f"Number of labels in y: {len(y)}")
    print(f"Target values detected: {sorted(list(set(y)))}")

    # --- TASK 3: Train/Test Split ---
    print("\n=== Machine Learning: Train/Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # --- TASK 4: Model Training ---
    print("\n=== Machine Learning: Model Training ===")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    model_name = type(model).__name__
    print(f"Model: {model_name}")
    print("Training completed successfully.")

    # --- TASK 5: Prediction ---
    print("\n=== Machine Learning: Prediction ===")
    predictions = model.predict(X_test)
    print("Predictions generated for test set.")
    print(f"Number of predictions: {len(predictions)}")
    example_predictions = [int(p) for p in predictions[:5]]
    print(f"Example predictions: {example_predictions}")

    # --- TASK 6: Evaluation ---
    print("\n=== Machine Learning: Evaluation ===")
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # --- TASK 7: Saving and Inspecting Model ---
    print("\n=== Machine Learning: Saving and Inspecting Model ===")
    joblib.dump(model, model_output_path)
    print(f"Saved model: {model_output_path}")
    print(f"Model type: {model_name}")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    
    print("\nDecision Tree Rules:")
    tree_rules = export_text(model, feature_names=column_names)
    print(tree_rules)

    # --- TASK 8: Saving Evaluation Results ---
    print("\n=== Machine Learning: Saving Evaluation Results ===")
    with open(evaluation_output_path, 'w', encoding='utf-8') as f:
        f.write("OOAIS Model Evaluation\n")
        f.write("======================\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
    print(f"Saved file: {evaluation_output_path}")

    # --- TASK 9: Saving Training Report ---
    print("\n=== Machine Learning: Saving Training Report ===")
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write("OOAIS Model Training Summary\n")
        f.write("============================\n\n")
        f.write("Input datasets\n")
        f.write("--------------\n")
        f.write(f"{features_path}\n")
        f.write(f"{labels_path}\n\n")
        f.write("Dataset statistics\n")
        f.write("------------------\n")
        f.write(f"Number of samples: {len(X)}\n")
        f.write(f"Number of features: {len(column_names)}\n\n")
        f.write("Model\n")
        f.write("-----\n")
        f.write(f"{model_name}\n\n")
        f.write("Train/Test split\n")
        f.write("----------------\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write("Evaluation summary\n")
        f.write("------------------\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
    print(f"Saved file: {report_output_path}")

if __name__ == "__main__":
    run_model_training()