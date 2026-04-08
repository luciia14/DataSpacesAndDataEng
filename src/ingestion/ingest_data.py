import json
import csv
import os

def save_csv(data, fieldnames, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def run_ingestion():
    # Rutas
    metadata_path = "data/raw/metadata.json"
    dataset_path = "data/raw/orbital_observations.csv"
    valid_output = "data/processed/observations_valid.csv"
    invalid_output = "data/processed/observations_invalid.csv"
    model_input_output = "data/processed/model_input.csv"
    report_path = "reports/ingestion_summary.txt"

   
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    rows = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        dataset_columns = reader.fieldnames
        for row in reader:
            rows.append(row)

   
    feature_cols = metadata.get('feature_columns', [])
    target_col = metadata.get('target_column', '')
    
    # Verificar si todas las features existen en el dataset real
    missing_features = [f for f in feature_cols if f not in dataset_columns]
    feature_status = "OK" if not missing_features else f"MISSING: {missing_features}"
    
    # Verificar si el target existe
    target_status = "OK" if target_col in dataset_columns else f"MISSING: {target_col}"

    
    col_validation = "OK" if dataset_columns == metadata['columns'] else "MISMATCH"
    record_count_validation = "OK" if len(rows) == metadata['num_records'] else "MISMATCH"

   
    valid_records = []
    invalid_records = []
    for record in rows:
        if record['temperature'].upper() == "INVALID":
            invalid_records.append(record)
        else:
            valid_records.append(record)

   
    model_input_data = []
    for record in valid_records:
        filtered_row = {key: record[key] for key in feature_cols if key in record}
        model_input_data.append(filtered_row)

    
    save_csv(valid_records, dataset_columns, valid_output)
    save_csv(invalid_records, dataset_columns, invalid_output)
    save_csv(model_input_data, feature_cols, model_input_output)

    
    summary_content = f"""Dataset: {metadata['dataset_name']}
Records loaded: {len(rows)}
Expected records: {metadata['num_records']}
Column validation: {col_validation}
Record count validation: {record_count_validation}
Feature validation: {feature_status}
Target validation: {target_status}
Valid records: {len(valid_records)}
Invalid records: {len(invalid_records)}
Generated files:
- {valid_output}
- {invalid_output}
- {model_input_output}
"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(summary_content)

    print(summary_content)

if __name__ == "__main__":
    run_ingestion()
