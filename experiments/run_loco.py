from src.dataset import BBC021Dataset
from src.preprocessing import create_dataloaders
from src.splitting import create_validation_split, create_loco_folds
from src.model import MoAResNet
from src.train import train_model
from src.evaluate import get_predictions, calculate_metrics
from src.utils import total_rss_gb
import torch
import pandas as pd
from pathlib import Path

def main():
    verbose = True
    SAVE_BEST_MODEL = True

    NUM_FOLDS = 1
    NUM_EPOCHS = 1

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    IMAGE_CSV_PATH = Path(__file__).resolve().parent.parent/"data"/"raw"/"BBBC021_v1_image.csv"
    MOA_CSV_PATH = Path(__file__).resolve().parent.parent/"data"/"raw"/"BBBC021_v1_moa.csv"
    IMAGE_ROOT_PATH = Path(__file__).resolve().parent.parent/"data"/"raw"/"images"
    MODEL_PATH = Path(__file__).resolve().parent.parent/"models"

    dataset = BBC021Dataset(
        image_csv=IMAGE_CSV_PATH,
        moa_csv=MOA_CSV_PATH,
        image_root=IMAGE_ROOT_PATH
    )

    loco_folds = create_loco_folds(dataset)

    results = []

    for fold in range(NUM_FOLDS):
        if verbose:
            print("\n" + "="*70)
            print(f"LOCO FOLD {fold + 1}/{NUM_FOLDS}")
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
            test_indices=test_indices
        )

        model = MoAResNet()

        save_name = f"loco_fold_V1_{fold+1:02d}.pth"

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            save_best_model=SAVE_BEST_MODEL,
            save_name=save_name
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

        metrics = calculate_metrics(predictions, labels,
                                    label_mapping=dataset.label_mapping)

        results.append({
            "fold": fold + 1,
            "test_compound": test_compound,
            "test_moa": dataset.metadata.iloc[test_indices]["moa"].unique(),
            "n_train_images": len(train_indices),
            "n_val_images": len(val_indices),
            "n_test_images": len(test_indices),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"]
        })

        if fold < NUM_FOLDS-1:
            del model, train_loader, val_loader, test_loader
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        MODEL_PATH / "loco_fold_V1_results.csv",
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

if __name__ == "__main__":
    main()
