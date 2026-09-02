import json
import random

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import psutil
import os


DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent / "mappings" / "moa_label_map.json"

def create_moa_label_mapping(df, label_column="moa", save_path=None):
    """Create and save a mapping from MoA labels to integer labels."""
    if save_path is None:
        save_path = DEFAULT_MAPPING_PATH
    save_path = Path(save_path)

    le = LabelEncoder()

    mapping = {
        label: int(index)
        for index, label in enumerate(le.fit(df[label_column]).classes_)
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(mapping, f, indent=2)

    return mapping

def load_moa_label_mapping(mapping_path=None):
    """Load MoA-ti-integer label mapping from json."""
    if mapping_path is None:
        mapping_path = DEFAULT_MAPPING_PATH
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Label mapping not found at {mapping_path}"
        )

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    return mapping

def decode_moa_label(label):
    """Convert integer label back to its MoA label."""
    mapping = load_moa_label_mapping()
    label_to_moa = {v: k for k, v in mapping.items()}
    return label_to_moa[label]

def calculate_class_weights(dataset, train_indices):
    labels = [
        dataset.label_mapping[dataset.metadata.iloc[i]["moa"]]
        for i in train_indices
    ]

    labels = np.array(labels)

    classes = np.arange(len(dataset.label_mapping))

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )

    return torch.tensor(class_weights, dtype=torch.float32)

def total_rss_gb():
    proc = psutil.Process(os.getpid())
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 1024**3

def set_seed(seed=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Set seed for reproducibility in Dataloader.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
