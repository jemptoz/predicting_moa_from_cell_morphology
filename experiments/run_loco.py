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
from src.evaluate import get_predictions, calculate_metrics
from src.utils import total_rss_gb, set_seed
import torch
import pandas as pd
import os

def run_loco(experiment_name, num_folds, num_epochs, seed, verbose=None, save_best_model= None, device=None):
    """

    """
    EXPERIMENT_NAME = experiment_name
    NUM_FOLDS = num_folds
    NUM_EPOCHS = num_epochs

    if verbose is None:
        verbose = False

    if save_best_model is None:
        SAVE_BEST_MODEL = True
    else: SAVE_BEST_MODEL = save_best_model

    if device is None:
        if torch.cuda.is_available():
            DEVICE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"
    else: DEVICE = device

    IMAGE_CSV_PATH = Path(__file__).resolve().parent.parent/"data"/"raw"/"BBBC021_v1_image.csv"
    MOA_CSV_PATH = Path(__file__).resolve().parent.parent/"data"/"raw"/"BBBC021_v1_moa.csv"
    IMAGE_ROOT_PATH = Path(__file__).resolve().parent.parent/"data"/"raw"/"images"

    MODEL_PATH = Path(__file__).resolve().parent.parent/"models"/EXPERIMENT_NAME
    os.makedirs(MODEL_PATH, exist_ok=True)
    RESULTS_PATH = Path(__file__).resolve().parent.parent/"results"/EXPERIMENT_NAME
    os.makedirs(RESULTS_PATH, exist_ok=True)

    set_seed(seed=seed)

    dataset = BBC021Dataset(
        image_csv=IMAGE_CSV_PATH,
        moa_csv=MOA_CSV_PATH,
        image_root=IMAGE_ROOT_PATH
    )

    loco_folds = create_loco_folds(dataset)

    results = []
    split_information = []

    for fold in range(NUM_FOLDS):
        if verbose:
            print("\n" + "="*70)
            print(f"{EXPERIMENT_NAME} FOLD: {fold + 1}/{NUM_FOLDS}")
            total_rss = total_rss_gb()
            print(f" Total RSS memory: {total_rss:.2f} GB | "
                  f" MPS allocated: {torch.mps.current_allocated_memory()/1024**3:.2f} GB | "
                  f" MPS driver reserved: {torch.mps.driver_allocated_memory()/1024**3:.2f} GB")
            print("="*70)

        dev_indices, test_indices, test_compound = next(loco_folds)

        train_indices, val_indices = create_validation_split(dataset, dev_indices)

        if verbose:
            print(f"Train images: {len(train_indices)}")
            print(f"Validation images: {len(val_indices)}")
            print(f"Test images: {len(test_indices)}")

        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            seed=seed
        )

        model = MoAResNet()

        save_name = f"fold_{fold+1:02d}.pth"

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            save_best_model=SAVE_BEST_MODEL,
            save_dir=MODEL_PATH,
            save_name=save_name,
            verbose=verbose
        )

        checkpoint = torch.load(
            MODEL_PATH / save_name,
            map_location="cpu"
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model.to(DEVICE)

        predictions, labels = get_predictions(
            model, test_loader, DEVICE
        )

        metrics = calculate_metrics(labels, predictions,
                                    label_mapping=dataset.label_mapping)

        results.append({
            "exp_name": EXPERIMENT_NAME,
            "fold": fold + 1,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"]
        })

        split_information.append({
            "exp_name": EXPERIMENT_NAME,
            "fold": fold + 1,
            "test_compound": test_compound,
            "test_moa": dataset.metadata.iloc[test_indices]["moa"].unique(),
            "n_test_images": len(test_indices),
            "train_compounds": dataset.metadata.iloc[train_indices]["Image_Metadata_Compound"].unique(),
            "train_moa": dataset.metadata.iloc[train_indices]["moa"].unique(),
            "n_train_images": len(train_indices),
            "val_compounds": dataset.metadata.iloc[val_indices]["Image_Metadata_Compound"].unique(),
            "val_moa": dataset.metadata.iloc[val_indices]["moa"].unique(),
            "n_val_images": len(val_indices)
        })


        if fold < NUM_FOLDS-1:
            del model, train_loader, val_loader, test_loader
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()


    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULTS_PATH / "results.csv",
        index=False
    )

    split_info_df = pd.DataFrame(split_information)
    split_info_df.to_csv(
        RESULTS_PATH / "split_information.csv",
        index=False
    )

    if verbose:
        print("\n" + "="*70)
        print("FINAL LOCO RESULTS")
        print("="*70)
        print("\nSummary: ")
        print(
            f"Accuracy: "
            f"{results_df.accuracy.mean():.3f} ± "
            f"{results_df.accuracy.std():.3f}"
        )
        print(
            f"Macro F1: "
            f"{results_df.macro_f1.mean():.3f} ± "
            f"{results_df.macro_f1.std():.3f}"
        )
        print(
            f"Weighted F1: "
            f"{results_df.weighted_f1.mean():.3f} ± "
            f"{results_df.weighted_f1.std():.3f}"
        )


def main():
    #parser = argparse.ArgumentParser()
    run_loco(
        experiment_name="loco_V1",
        num_folds=2,
        num_epochs=2,
        seed=42,
        verbose=True
    )

if __name__ == "__main__":
    main()
