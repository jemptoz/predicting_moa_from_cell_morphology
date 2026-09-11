import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
from src.utils import calculate_class_weights
from sklearn.metrics import f1_score
from pathlib import Path
from copy import deepcopy
from time import perf_counter
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    loss_denominator = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_denominator = (
            criterion.weight[labels].sum().item()
            if criterion.weight is not None
            else labels.numel()
        )
        running_loss += loss.item() * batch_denominator
        loss_denominator += batch_denominator

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)

    epoch_f1 = f1_score(
        all_labels,
        all_preds,
        labels=np.unique(all_labels),
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_f1

def val_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    loss_denominator = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_denominator = (
                criterion.weight[labels].sum().item()
                if criterion.weight is not None
                else labels.numel()
            )
            running_loss += loss.item() * batch_denominator
            loss_denominator += batch_denominator

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        if loss_denominator == 0:
            raise ValueError("Cannot calculate loss for an empty dataloader.")
        epoch_loss = running_loss / loss_denominator

        epoch_f1 = f1_score(
            all_labels,
            all_preds,
            labels=np.unique(all_labels),
            average="macro",
            zero_division=0
        )

        return epoch_loss, epoch_f1

def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        device,
        save_dir,
        save_name,
        learning_rate=1e-4,
        weight_decay=1e-4,
        early_stopping_patience=8,
        scheduler_patience=3,
        verbose=False,
        save_best_model=True
):

    model.to(device)

    class_weights = calculate_class_weights(
        train_loader.dataset.dataset,
        train_loader.dataset.indices
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=scheduler_patience
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_macro_f1": [],
        "learning_rate": [],
        "epoch_seconds": [],
    }

    best_val_f1 = float("-inf")
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_start = perf_counter()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]

        train_loss, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_f1 = val_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_macro_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["learning_rate"].append(epoch_learning_rate)
        history["epoch_seconds"].append(perf_counter() - epoch_start)

        if verbose:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train Macro F1: {train_f1:.3f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val Macro F1: {val_f1:.3f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            if save_best_model:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_macro_f1': val_f1,
                    'val_loss': val_loss
                }, os.path.join(save_dir, save_name))
                if verbose: print(f"Best model saved (val_f1: {val_f1:.4f})")
            else:
                best_state = deepcopy({
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                })

        else :
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            if verbose: print(f"Early stopping after {epoch + 1} epochs")
            break

    if not save_best_model:
        model.load_state_dict(best_state)

    return history

