from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt

def ordered_classes(label_mapping):
    """

    """
    items = sorted(label_mapping.items(), key=lambda x: x[1])
    class_ids = [int(label) for _, label in items]

    if class_ids != list(range(len(items))):
        raise ValueError("Class labels are not ordered.")

    return [name for name,_ in items]

def get_predictions(model, dataloader, device, return_probability=False):
    """
    Generate predictions and collect true labels.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataloader : DataLoader
        Dataloader containing the dataset to evaluate.
    device : torch.device
        Device on which the model is evaluated.

    Returns
    -------
    predictions : np.ndarray
        Predicted class labels.
    labels : np.ndarray
        True class labels.

    """
    model.eval()
    all_labels = []
    all_probabilities = []

    with torch.inference_mode():
        for images, labels in dataloader:
            logits = model(images.to(device))

            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite logits.")

            probabilities = torch.softmax(logits.float(), dim=1)
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        if not all_labels:
            raise ValueError("Cannot evaluate an empty dataloader.")

        labels = np.concatenate(all_labels).astype(np.int64)
        probabilities = np.concatenate(all_probabilities)
        predictions = np.argmax(probabilities, axis=1)

        if return_probability:
            return predictions, labels, probabilities
        else:
            return predictions, labels

    # with torch.no_grad():
    #     for images, labels in dataloader:
    #
    #         images = images.to(device)
    #
    #         outputs = model(images)
    #         predictions = torch.argmax(outputs, dim=1)
    #
    #         all_predictions.extend(predictions.cpu().numpy())
    #         all_labels.extend(labels.numpy())
    #
    # return np.array(all_predictions), np.array(all_labels)

def calculate_metrics(labels, predictions, label_mapping):
    """

    """
    class_names = ordered_classes(label_mapping)
    class_ids = np.arange(len(class_names))
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        zero_division=0
    )
    supported = support > 0

    summary = {
        "n_images": int(labels.size),
        "n_classes_supported": int(supported.sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(recall[supported].mean()),
        "macro_f1": float(f1.mean()),
        "macro_f1_supported": float(f1[supported].mean()),
        "weighted_f1": float(np.average(f1, weights=support))
    }

    per_class = pd.DataFrame({
        "class_id": class_ids,
        "moa": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
        "predicted_support": np.bincount(
            predictions, minlength=len(class_names)
        ),
    })

    cm = confusion_matrix(labels, predictions, labels=class_ids)

    return {
        "summary": summary,
        "per_class": per_class,
        "confusion_matrix": cm,
    }

def save_evaluation(labels, predictions, label_mapping, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = calculate_metrics(labels, predictions, label_mapping)
    class_names = ordered_classes(label_mapping)
    cm = metrics["confusion_matrix"]

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics["summary"], f, indent=2, allow_nan=False)

    metrics["per_class"].to_csv(output_dir / "per_class.csv", index=False)

    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(class_names, name="true_moa"),
        columns=pd.Index(class_names, name="predicted_moa")
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted MoA")
    ax.set_ylabel("True MoA")
    ax.set_title("MoA confusion matrix")

    threshold = cm.max() / 2
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            ax.text(
                col, row, str(cm[row, col]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    return metrics

def make_prediction_table(
        dataset,
        test_indices,
        labels,
        predictions,
        probabilities,
        fold,
        test_compound
):
    class_names = ordered_classes(dataset.label_mapping)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    #This requires shuffle=False and drop_last=False on test_loader(!)
    metadata = dataset.metadata.iloc[list(test_indices)].copy()
    expected_labels = metadata["moa"].map(dataset.label_mapping).to_numpy()

    table = metadata.reset_index(drop=True)
    table.insert(0, "dataset_index", list(test_indices))
    table.insert(0, "fold", fold)
    table["test_compound"] = test_compound
    table["true_label"] = labels
    table["predicted_label"] = predictions
    table["true_moa"] = [class_names[int(label)] for label in labels]
    table["predicted_moa"] = [
        class_names[int(label)] for label in predictions
    ]
    table["correct"] = labels == predictions

    sorted_probabilities = np.sort(probabilities, axis=1)
    confidence = sorted_probabilities[:, -1]
    second_probability = (
        sorted_probabilities[:, -2]
        if len(class_names) > 1
        else np.zeros(len(labels))
    )

    safe_probabilities = np.clip(
        probabilities, np.finfo(np.float64).tiny, 1.0
    )
    entropy = -np.sum(
        probabilities * np.log(safe_probabilities), axis=1
    )
    true_probability = probabilities[np.arange(len(labels)), labels]

    table["confidence"] = confidence
    table["second_highest_probability"] = second_probability
    table["probability_margin"] = confidence - second_probability
    table["entropy"] = entropy
    table["normalised_entropy"] = (
        entropy / np.log(len(class_names))
        if len(class_names) > 1
        else np.zeros(len(labels))
    )
    table["true_class_probability"] = true_probability
    table["negative_log_likelihood"] = -np.log(
        np.clip(true_probability, np.finfo(np.float64).tiny, 1.0)
    )

    for class_id in range(len(class_names)):
        table[f"prob_class_{class_id}"] = probabilities[:, class_id]

    return table
