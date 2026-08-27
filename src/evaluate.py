import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

def get_predictions(model, dataloader, device):
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

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:

            images = images.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_predictions), np.array(all_labels)

def calculate_metrics(labels, predictions, label_mapping):
    """
    Calculate classification performance metrics.

    Parameters
    ----------
    labels : np.ndarray
        True class labels.
    predictions : np.ndarray
        Predicted class labels.
    label_mapping : dict
        Mapping from MoA names to integer labels.

    Returns
    -------
    metrics : dict
        Accuracy and F1 scores
    """
    accuracy = accuracy_score(labels, predictions)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    class_names = [
        moa for moa, label in sorted(label_mapping.items(), key=lambda x: x[1])
    ]

    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report
    }

def plot_confusion_matrix(labels, predictions, label_mapping):
    """
    Plot confusion matrix.

    Parameters
    ----------
    labels : np.ndarray
        True class labels.
    predictions : np.ndarray
        Predicted class labels.
    label_mapping : dict
        Mapping from MoA names to integer labels.
    """
    class_names = [
        moa for moa, label in sorted(label_mapping.items(), key=lambda x: x[1])
    ]

    cm = confusion_matrix(
        labels,
        predictions,
        labels=range(len(class_names))
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(
        class_names,
        rotation=90
    )

    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted MoA")
    ax.set_ylabel("True MoA")
    ax.set_title("MoA Classification Confusion Matrix")

    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.show()