import csv
import os
import json
from datetime import datetime

def save_csv(data, fieldnames, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def prepare_ml_input():
    input_file = "data/processed/observations_valid.csv"
    features_output = "data/processed/model_features.csv"
    labels_output = "data/processed/model_labels.csv"
    
    
    base_numeric_columns = ['temperature', 'velocity', 'altitude', 'signal_strength']
    final_features = [
        'temperature', 'velocity', 'altitude', 'signal_strength',
        'temperature_velocity_interaction', 'altitude_signal_ratio', 'hour_normalized'
    ]
    
    accepted_records = []
    rejected_count = 0
    total_loaded = 0

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run ingestion first.")
        return

    # TASK 1: Loading and Conversion
    print("=== ML Input Preparation: Loading and Conversion ===")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_loaded += 1
            try:
                for col in base_numeric_columns:
                    val = float(row[col])
                    if col == 'altitude' and val < 0:
                        raise ValueError
                    row[col] = val
                accepted_records.append(row)
            except (ValueError, TypeError):
                rejected_count += 1

    print(f"Input file: {input_file}")
    print(f"Records loaded: {total_loaded}")
    print(f"Records accepted: {len(accepted_records)}")
    print(f"Records rejected: {rejected_count}")

    # TASK 2: Normalization
    print("\n=== ML Input Preparation: Normalization ===")
    for col in base_numeric_columns:
        values = [r[col] for r in accepted_records]
        col_min, col_max = min(values), max(values)
        if col_max - col_min != 0:
            for r in accepted_records:
                r[col] = round((r[col] - col_min) / (col_max - col_min), 4)
        else:
            for r in accepted_records:
                r[col] = 0.0
    print("Normalization completed successfully.")

    # TASK 3: Derived Features 
    print("\n=== ML Input Preparation: Derived Features ===")
    for r in accepted_records:
        r['temperature_velocity_interaction'] = round(r['temperature'] * r['velocity'], 4)
        r['altitude_signal_ratio'] = round(r['altitude'] / (r['signal_strength'] + 0.0001), 4)
    print("New features added: interaction and ratio.")

    # TASK 4: Temporal Features 
    print("\n=== ML Input Preparation: Temporal Features ===")
    for r in accepted_records:
        dt = datetime.fromisoformat(r['timestamp'])
        r['hour_normalized'] = round(dt.hour / 24.0, 4)
    
    print("New feature added: hour_normalized")
    print("\nExample record (extended):")
    print(json.dumps(accepted_records[0], indent=2))

    # TASK 5: Feature Selection 
    print("\n=== ML Input Preparation: Feature Selection ===")
    feature_dataset = []
    label_dataset = []

    for r in accepted_records:
        # Extraer solo las features seleccionadas
        feature_row = {feature: r[feature] for feature in final_features}
        feature_dataset.append(feature_row)
        
        # Extraer el target (label)
        label_row = {'anomaly_flag': r['anomaly_flag']}
        label_dataset.append(label_row)

    print("Selected features:")
    for feature in final_features:
        print(f"- {feature}")
    print("\nExample record (final):")
    print(json.dumps(feature_dataset[0], indent=2))

    # TASK 6: Saving Outputs 
    print("\n=== ML Input Preparation: Saving Outputs ===")
    save_csv(feature_dataset, final_features, features_output)
    save_csv(label_dataset, ['anomaly_flag'], labels_output)

    print(f"Saved file: {features_output}")
    print(f"Saved file: {labels_output}")
    print(f"Number of records: {len(feature_dataset)}")
    print(f"Number of features: {len(final_features)}")
    
    print("\nExample label record:")
    print(json.dumps(label_dataset[0], indent=2))

if __name__ == "__main__":
    prepare_ml_input()