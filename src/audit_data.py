import pandas as pd
import numpy as np
import os
import json
from src.data_loader import DataLoader

def audit_dataset(module_id):
    """Performs comprehensive auditing of the dataset."""
    try:
        df = DataLoader.load_dataset(module_id)
        is_valid, missing_cols = DataLoader.validate_schema(df, module_id)

        audit_report = {
            'module_id': module_id,
            'dimensions': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'data_types': df.dtypes.astype(str).to_dict(),
            'target_distribution': df[DataLoader.DATA_CONFIG[module_id]['target']].value_counts().to_dict() if is_valid else "Schema invalid",
            'schema_valid': is_valid,
            'missing_columns': missing_cols
        }

        # Save report
        report_path = f"reports/dataset_audits/{module_id}_audit.json"
        with open(report_path, 'w') as f:
            json.dump(audit_report, f, indent=4)

        return audit_report
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    for module in DataLoader.DATA_CONFIG.keys():
        print(f"Auditing {module}...")
        result = audit_dataset(module)
        print(result)
