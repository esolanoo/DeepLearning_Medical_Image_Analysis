import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef)

from funcs import *
from preprocessing import data_loaders
from models import build_model, count_parameters


EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 1e-3
NUM_CLASSES = 8

# For local run
CHECKPOINT_DIR = GLOBAL_DIR + r"src/checkpoints"
RESULT_DIR = GLOBAL_DIR + r"src/results"

# For Colab
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = PROJECT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["resnet18", "efficientnet_b0", "vit_b_16", "small_cnn", "alexnet"]

set_env()

# ---------------- Data ----------------
data = data_loaders()
train_loader, val_loader, test_loader = [loader for loader in data['loaders']]
train_ds, val_ds, test_ds = [ds for ds in data['datasets']]


# ---------------- Metrics ----------------
def calculate_metrics(y_true, y_pred):
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    specificity_per_class = []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        denominator = tn + fp
        specificity_per_class.append(
            tn / denominator if denominator else float("nan")
        )
        
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_specificity": float(np.nanmean(specificity_per_class)),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    return results


# ---------------- Evaluation ----------------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ---------------- Training ----------------
def train_one_model(name):
    print(f"\nTraining: {name} for {EPOCHS} epochs\n")
    model = build_model(name, num_classes=NUM_CLASSES).to(DEVICE)
    counts = count_parameters(model)
    print("Parameters:", counts)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=1e-4,)

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    history = []

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
        
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Keep frozen CNN BatchNorm layers in evaluation mode.
        # This avoids updating running statistics in frozen backbones.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()

        running_loss = 0.0
        y_train_true, y_train_pred = [], []
    
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
        )
        
        for images, labels in progress_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            y_train_true.extend(labels.detach().cpu().tolist())
            y_train_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{running_loss / len(y_train_true):.4f}",
            )

        progress_bar.close()
        train_loss = running_loss / len(train_ds)
        train_metrics = calculate_metrics(y_train_true, y_train_pred)
        val_metrics = evaluate(model, val_loader, criterion)

        row = {
            "model": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_specificity": val_metrics["macro_specificity"],
            "val_mcc": val_metrics["mcc"],
        }
        history.append(row)
        """
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val acc={val_metrics['accuracy']:.4f} | "
            f"val F1={val_metrics['macro_f1']:.4f} | "
            f"val spec={val_metrics['macro_specificity']:.4f} | "
            f"val MCC={val_metrics['mcc']:.4f}"
        )
        """

        # Select the checkpoint by validation macro-F1.
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_name": name,
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "class_to_idx": train_ds.df[["class_name", "label"]].drop_duplicates().set_index("class_name")["label"].to_dict(),
                    "validation_metrics": val_metrics,
                },
                CHECKPOINT_DIR / f"{name}_best.pth",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    elapsed = time.perf_counter() - start_time
    peak_memory_mb = (torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) if DEVICE == "cuda" else None)

    pd.DataFrame(history).to_csv(RESULT_DIR / f"{name}_history.csv", index=False)

    summary = {
        "model": name,
        "total_parameters": counts["total"],
        "trainable_parameters": counts["trainable"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "epochs_completed": len(history),
        "training_seconds": int(elapsed),
        "peak_gpu_memory_mb": peak_memory_mb,
    }

    print("Summary:", summary)
    
    return summary


if __name__ == "__main__":
    summaries = []

    assert torch.cuda.is_available(), "GPU unavailable: check Colab runtime settings"

    print("Training device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(0))

    for model_name in MODELS:
        summaries.append(train_one_model(model_name))

    pd.DataFrame(summaries).to_csv(RESULT_DIR / f"training_summary.csv", index=False)
    print("\nTraining complete. Summary saved to results/")