import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- TASK 1: VALIDATION ---
def validate_input_files():
    required_files = ["data/processed/model_features.csv", "data/processed/model_labels.csv"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print("Error: missing required input file(s):")
        for f in missing_files:
            print(f"- {f}")
        raise SystemExit(1)

# --- TASK 2: LOADING ---
def load_data():
    features_path = "data/processed/model_features.csv"
    labels_path = "data/processed/model_labels.csv"
    print("\n=== Model Playground: Loading Data ===")
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    print(f"Feature file: {features_path}")
    print(f"Label file: {labels_path}")
    return features_df, labels_df

# --- TASK 3: INSPECTION ---
def inspect_data(features_df, labels_df):
    print("\n=== Model Playground: Data Inspection ===")
    target_values = [int(v) for v in sorted(labels_df["anomaly_flag"].unique())]
    print(f"Number of samples: {len(features_df)}")
    print(f"Number of features: {features_df.shape[1]}")
    print(f"Feature columns: {list(features_df.columns)}")
    print(f"Target values detected: {target_values}")

# --- TASK 4: PREPARATION ---
def prepare_features_and_labels(features_df, labels_df):
    print("\n=== Model Playground: Preparing Features and Labels ===")
    X = features_df.values
    y = labels_df["anomaly_flag"].astype(int).values
    print(f"X shape: {X.shape}") # Imprime (470, 7)
    print(f"y shape: {y.shape}") # Imprime (470,)
    return X, y

# --- TASK 5: SPLIT ---
def split_data(X, y):
    print("\n=== Model Playground: Train/Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# --- TASK 6: DEFINE MODELS ---
def define_models():
    return {
        "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

# --- TASK 7: TRAIN ---
def train_models(models, X_train, y_train):
    print("\n=== Model Playground: Training Models ===")
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name}: trained")
    return trained_models

# --- TASK 8: PREDICT ---
def generate_predictions(trained_models, X_test):
    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        results.append({"name": name, "model": model, "y_pred": y_pred})
    return results

# --- TASK 9: EXAMPLE PREDICTIONS (PRINT EXACTO) ---
def print_example_predictions(prediction_results, y_test, num_examples=5):
    print("\n=== Model Playground: Example Predictions ===")
    for i in range(num_examples):
        line = f"True: {y_test[i]}"
        for res in prediction_results:
            line += f" | {res['name']}: {res['y_pred'][i]}"
        print(line)

# --- TASK 10: ACCURACY ---
def compute_accuracy(prediction_results, y_test):
    print("\n=== Model Playground: Accuracy Comparison ===")
    for result in prediction_results:
        acc = accuracy_score(y_test, result["y_pred"])
        result["accuracy"] = acc
        print(f"{result['name']}: {acc:.4f}")
    return prediction_results

# --- TASK 11: DETAILED EVALUATION (PRINT EXACTO) ---
def compute_detailed_metrics(prediction_results, y_test):
    print("\n=== Model Playground: Detailed Evaluation ===")
    for result in prediction_results:
        y_pred = result["y_pred"]
        report = classification_report(y_test, y_pred, output_dict=True)
        result["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        result["classification_report"] = report
        
        print(f"Model: {result['name']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(result["confusion_matrix"])
        print("\nClass labels:")
        print("0 -> normal observation")
        print("1 -> anomaly")
        print("\nClassification Report:")
        print("-" * 60)
        print("Class           Precision Recall    F1-score  Support")
        print("-" * 60)
        
        # Filas de clases
        for label, name in [('0', '0 (normal)'), ('1', '1 (anomaly)')]:
            print(f"{name:<15} {report[label]['precision']:.2f}      {report[label]['recall']:.2f}      {report[label]['f1-score']:.2f}      {int(report[label]['support'])}")
        
        print("-" * 60)
        # Promedios
        print(f"Macro average   {report['macro avg']['precision']:.2f}      {report['macro avg']['recall']:.2f}      {report['macro avg']['f1-score']:.2f}      {int(report['macro avg']['support'])}")
        print(f"Weighted average {report['weighted avg']['precision']:.2f}      {report['weighted avg']['recall']:.2f}      {report['weighted avg']['f1-score']:.2f}      {int(report['weighted avg']['support'])}\n")
        
    return prediction_results

# --- TASK 12: RANKING ---
def rank_models(evaluation_results):
    print("\n=== Model Playground: Ranking ===")
    sorted_results = sorted(evaluation_results, key=lambda x: x["accuracy"], reverse=True)
    for index, result in enumerate(sorted_results, start=1):
        print(f"{index}. {result['name']} - {result['accuracy']:.4f}")
    return sorted_results

# --- TASK 13: EXPERIMENTS ---
def run_experiments(X_train, X_test, y_train, y_test):
    print("\n=== Model Playground: Controlled Experiments ===")
    exp_results = []
    for d in [2, 3, 5]:
        m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        exp_results.append({"name": f"Decision Tree (max_depth={d})", "accuracy": acc})
        print(f"Decision Tree (max_depth={d}): {acc:.2f}")
    return exp_results

# --- TASK 14: SAVING SUMMARY & VISUALIZATIONS ---
def save_experiment_summary(f_path, l_path, X, X_train, X_test, ranked_models, exp_results):
    print("\n=== Model Playground: Saving Summary ===")
    Path("reports").mkdir(exist_ok=True)
    report_file = "reports/model_playground_summary.txt"
    with open(report_file, "w") as f:
        f.write("OOAIS Model Playground Summary\n================================\n\n")
        f.write(f"Input datasets\n--------------\n{f_path}\n{l_path}\n\n")
        f.write(f"Dataset statistics\n------------------\nNumber of samples: {X.shape[0]}\nNumber of features: {X.shape[1]}\n")
        f.write(f"Training samples: {len(X_train)}\nTesting samples: {len(X_test)}\n\n")
        f.write("Compared models\n---------------\n")
        for res in ranked_models:
            f.write(f"- {res['name']}: {res['accuracy']:.4f}\n")
        f.write(f"\nBest model\n----------\n{ranked_models[0]['name']} achieved the highest accuracy: {ranked_models[0]['accuracy']:.4f}\n")
        f.write("\nConclusion\n----------\n")
        f.write(f"The best candidate for further experiments is {ranked_models[0]['name']},\nbecause it achieved the highest accuracy on the current test set.\n")
    print(f"Saved file: {report_file}")

def create_metric_plots(ranked_models):
    print("\n=== Model Playground: Saving Visualizations ===")
    names = [r["name"] for r in ranked_models]
    accs = [r["accuracy"] for r in ranked_models]
    precs = [r["classification_report"]["1"]["precision"] for r in ranked_models]
    recs = [r["classification_report"]["1"]["recall"] for r in ranked_models]
    f1s = [r["classification_report"]["1"]["f1-score"] for r in ranked_models]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].bar(names, accs, color='skyblue')
    axes[0, 0].set_title("Accuracy")
    axes[0, 1].bar(names, precs, color='salmon')
    axes[0, 1].set_title("Precision (Anomaly)")
    axes[1, 0].bar(names, recs, color='lightgreen')
    axes[1, 0].set_title("Recall (Anomaly)")
    axes[1, 1].bar(names, f1s, color='violet')
    axes[1, 1].set_title("F1-score (Anomaly)")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig("reports/model_comparison_panel.png")
    print("Saved file: reports/model_comparison_panel.png")
    plt.close()

# --- MAIN ORCHESTRATION ---
def run_playground():
    validate_input_files()
    f_df, l_df = load_data()
    inspect_data(f_df, l_df)
    X, y = prepare_features_and_labels(f_df, l_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    models = define_models()
    trained = train_models(models, X_train, y_train)
    results = generate_predictions(trained, X_test)
    
    print_example_predictions(results, y_test)
    results = compute_accuracy(results, y_test)
    results = compute_detailed_metrics(results, y_test)
    ranked = rank_models(results)
    
    exps = run_experiments(X_train, X_test, y_train, y_test)
    save_experiment_summary("data/processed/model_features.csv", "data/processed/model_labels.csv", X, X_train, X_test, ranked, exps)
    create_metric_plots(ranked)

if __name__ == "__main__":
    run_playground()