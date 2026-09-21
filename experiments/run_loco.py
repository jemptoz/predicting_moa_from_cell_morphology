import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.dataset import BBC021Dataset
from src.preprocessing import create_dataloaders
from src.splitting import create_validation_split, create_loco_folds
from src.model import MoAResNet
from src.train import train_model
from src.evaluate import get_predictions, make_prediction_table, save_evaluation
from src.utils import set_seed
import torch
import pandas as pd
import os
import gc
import argparse
import json
import platform
from time import perf_counter
import numpy as np
import sklearn

def run_loco(experiment_name, num_folds, num_epochs, seed,
             verbose=None, save_best_model= None, device=None, resume=False
             ):
    """

    """
    #DataLoader
    batch_size = 32
    num_workers = 2
    #Model
    learning_rate = 1e-4
    weight_decay = 1e-4
    early_stopping_patience = 8
    scheduler_patience = 3

    if verbose is None:
        verbose = False

    save_best_model = True if save_best_model is None else save_best_model

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else: device = device

    BASE_DIR = Path(__file__).resolve().parent.parent

    IMAGE_CSV_PATH = BASE_DIR /"data"/"raw"/"BBBC021_v1_image.csv"
    MOA_CSV_PATH = BASE_DIR /"data"/"raw"/"BBBC021_v1_moa.csv"
    IMAGE_ROOT_PATH = BASE_DIR /"data"/"raw"/"images"

    MODEL_PATH = BASE_DIR /"models"/experiment_name
    RESULTS_PATH = BASE_DIR /"results"/experiment_name

    if resume:
        if not RESULTS_PATH.exists():
            raise FileNotFoundError(f"Cannot resume because {RESULTS_PATH} does not exist.")
    else:
        for directory in (MODEL_PATH, RESULTS_PATH):
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f"{directory} is not empty. Use a new experiment name or pass --resume.")

    os.makedirs(RESULTS_PATH, exist_ok=True)
    if save_best_model:
        os.makedirs(MODEL_PATH, exist_ok=True)

    set_seed(seed=seed)

    dataset = BBC021Dataset(
        image_csv=IMAGE_CSV_PATH,
        moa_csv=MOA_CSV_PATH,
        image_root=IMAGE_ROOT_PATH
    )

    loco_folds = create_loco_folds(dataset, num_folds, seed=seed)

    config = {
        "experiment_name": experiment_name,
        "seed": seed,
        "num_epochs": num_epochs,
        "num_folds": num_folds,
        "total_compounds": len(dataset.get_compounds()),
        "device": str(device),
        "save_best_model": save_best_model,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "scheduler_patience": scheduler_patience,
        "validation_ratio": 0.2,
        "min_training_compounds_per_moa": 2,
        "selection_metric": "validation macro F1 over supported classes",
        "macro_f1_definition": "mean F1 across all mapped classes",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    config_path = RESULTS_PATH / "config.json"

    if resume:
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot resume because {config_path} does not exist.")
        with config_path.open() as f:
            previous_config = json.load(f)

        resume_keys = [
            "experiment_name",
            "seed",
            "num_epochs",
            "num_folds",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "early_stopping_patience",
            "scheduler_patience"
        ]

        mismatches = {
            key: {"previous": previous_config.get(key), "current": config.get(key)}
            for key in resume_keys if previous_config.get(key) != config.get(key)
        }

        if mismatches:
            raise ValueError("The resume configuration does not match the original run.")

    else:
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

    fold_rows = []
    prediction_tables = []
    split_rows = []
    split_information = []

    for fold in range(num_folds):
        if verbose:
            print("\n" + "="*70)
            print(f"{experiment_name} FOLD: {fold + 1}/{num_folds}")
            print("="*70)

        FOLD_PATH = RESULTS_PATH / f"fold_{fold+1:02d}"
        FOLD_PATH.mkdir(parents=True, exist_ok=True)

        dev_indices, test_indices, test_compound = next(loco_folds)

        train_indices, val_indices = create_validation_split(dataset, dev_indices, seed=seed)

        train_compound_set = dataset.get_compound_set(train_indices)
        val_compound_set = dataset.get_compound_set(val_indices)
        test_compound_set = dataset.get_compound_set(test_indices)

        if (
            train_compound_set & test_compound_set
            or val_compound_set & test_compound_set
            or train_compound_set & val_compound_set
        ):
            raise ValueError(f"Compound leakage in fold {fold+1}.")

        if verbose:
            print(f"Train images: {len(train_indices)}")
            print(f"Validation images: {len(val_indices)}")
            print(f"Test images: {len(test_indices)}")

        split_details = {
            "fold": fold+1,
            "test_compound": str(test_compound)
        }
        for split_name, indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices)
        ):
            split_details[f"{split_name}_indices"] = [
                int(index) for index in indices
            ]
            split_details[f"{split_name}_compounds"] = sorted(
                str(value) for value in dataset.get_compound_set(indices)
            )
            split_details[f"{split_name}_moa"] = sorted(
                dataset.metadata.iloc[indices]["moa"].unique().tolist()
            )
            split_details[f"n_{split_name}_images"] = len(indices)

        split_rows.append(split_details)
        with (RESULTS_PATH / "split_information.json").open("w") as f:
            json.dump(split_rows, f, indent=2)

        split_information.append({
            "exp_name": experiment_name,
            "fold": fold + 1,
            "test_compound": test_compound,
            "test_moa": ", ".join(sorted(dataset.metadata.iloc[test_indices]["moa"].unique().tolist())),
            "n_test_images": len(test_indices),
            "train_compounds": ", ".join(sorted(dataset.metadata.iloc[train_indices]["Image_Metadata_Compound"].unique().tolist())),
            "train_moa": ", ".join(sorted(dataset.metadata.iloc[train_indices]["moa"].unique().tolist())),
            "n_train_images": len(train_indices),
            "val_compounds": ", ".join(sorted(dataset.metadata.iloc[val_indices]["Image_Metadata_Compound"].unique().tolist())),
            "val_moa": ", ".join(sorted(dataset.metadata.iloc[val_indices]["moa"].unique().tolist())),
            "n_val_images": len(val_indices)
        })

        required_fold_files = [
            "training_history.csv",
            "training_summary.json",
            "prediction_table.csv",
            "metrics.json",
            "per_class.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png"
        ]

        fold_is_complete = all(
            (FOLD_PATH / file).exists() for file in required_fold_files
        )

        if resume and fold_is_complete:
            if verbose:
                print(f"Fold {fold+1} is already complete.")

            table = pd.read_csv(FOLD_PATH/"prediction_table.csv")
            prediction_tables.append(table)

            with (FOLD_PATH / "metrics.json").open() as file:
                saved_metrics = json.load(file)

            with (FOLD_PATH / "training_summary.json").open() as file:
                training_summary = json.load(file)

            fold_rows.append({
                "exp_name": experiment_name,
                "fold": fold + 1,
                "test_compound": test_compound,
                **saved_metrics,
                **training_summary,
            })

            continue

        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed
        )

        model = MoAResNet()
        training_start = perf_counter()

        save_name = f"fold_{fold+1:02d}.pth"

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device,
            save_dir=MODEL_PATH,
            save_name=save_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            scheduler_patience=scheduler_patience,
            verbose=verbose,
            save_best_model=save_best_model
        )

        training_duration = perf_counter() - training_start
        history_df = pd.DataFrame(history)
        history_df.to_csv(FOLD_PATH / "training_history.csv", index=False)

        best_index = int(np.argmax(history["val_macro_f1"]))
        training_summary = {
            "best_epoch": int(history["epoch"][best_index]),
            "best_val_macro_f1": float(
                history["val_macro_f1"][best_index]
            ),
            "best_val_loss": float(history["val_loss"][best_index]),
            "epochs_completed": len(history["epoch"]),
            "stopped_early": len(history["epoch"]) < num_epochs,
            "training_duration": training_duration,
        }
        with (FOLD_PATH / f"training_summary.json").open("w") as file:
            json.dump(training_summary, file, indent=2, allow_nan=False)

        checkpoint = torch.load(
            MODEL_PATH / save_name,
            map_location="cpu"
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model.to(device)

        del checkpoint

        predictions, labels, probabilities = get_predictions(
            model, test_loader, device, return_probability=True
        )

        table = make_prediction_table(
            dataset=dataset,
            test_indices=test_indices,
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
            fold=fold+1,
            test_compound=test_compound
        )

        table.to_csv(FOLD_PATH / "prediction_table.csv", index=False)
        prediction_tables.append(table)

        metrics = save_evaluation(
            labels,
            predictions,
            dataset.label_mapping,
            FOLD_PATH
        )

        fold_rows.append({
            "exp_name": experiment_name,
            "fold": fold+1,
            "test_compound": test_compound,
            **metrics["summary"],
            **training_summary
        })
        pd.DataFrame(fold_rows).to_csv(
            RESULTS_PATH / "results.csv", index=False
        )

        del model, train_loader, val_loader, test_loader
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    split_info_df = pd.DataFrame(split_information)
    split_info_df.to_csv(
        RESULTS_PATH / "split_information.csv",
        index=False
    )

    pooled = pd.concat(prediction_tables, ignore_index=True)

    if pooled["dataset_index"].duplicated().any():
        raise ValueError("An image has multiple pooled test predictions.")

    POOLED_PATH = RESULTS_PATH / "pooled"
    POOLED_PATH.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(POOLED_PATH / "predictions.csv", index=False)

    pooled_metrics = save_evaluation(
        pooled["true_label"].to_numpy(),
        pooled["predicted_label"].to_numpy(),
        dataset.label_mapping,
        POOLED_PATH
    )

    confidence_columns = [
        "confidence",
        "probability_margin",
        "entropy",
        "normalised_entropy",
        "true_class_probability",
        "negative_log_likelihood",
    ]
    pooled[confidence_columns].describe().to_csv(
        POOLED_PATH / "confidence_summary.csv"
    )
    pooled.groupby("correct")[confidence_columns].agg(
        ["count", "mean", "std", "median"]
    ).to_csv(POOLED_PATH / "confidence_by_correctness.csv")

    metric_columns = [
        "accuracy",
        "macro_f1",
        "macro_f1_supported",
        "weighted_f1",
        "balanced_accuracy",
    ]

    pd.DataFrame(fold_rows)[metric_columns].agg(
        ["mean", "std"]
    ).to_csv(RESULTS_PATH / "fold_summary.csv")

    if verbose:
        print("\n" + "="*70)
        print("FIN.")
        print("="*70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    run_loco(
        experiment_name=args.experiment_name,
        num_folds=args.folds,
        num_epochs=args.epochs,
        seed=args.seed,
        verbose=args.verbose,
        resume=args.resume
    )

if __name__ == "__main__":
    main()
