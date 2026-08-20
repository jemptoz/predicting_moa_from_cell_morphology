import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def create_moa_label_mapping(df, label_column="moa", save_path="mappings/moa_label_map.json"):
    """Create and save a mapping from MoA labels to integer labels."""
    le = LabelEncoder()

    mapping = {
        label: int(index)
        for index, label in enumerate(le.fit(df[label_column]).classes_)
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(mapping, f, indent=2)

    return mapping

def load_moa_label_mapping(mapping_path="mappings/moa_label_map.json"):
    """Load MoA-ti-integer label mapping from json."""
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    return mapping

def decode_moa_label(label):
    """Convert integer label back to its MoA label."""
    mapping = load_moa_label_mapping()
    label_to_moa = {v: k for k, v in mapping.items()}
    return label_to_moa[label]