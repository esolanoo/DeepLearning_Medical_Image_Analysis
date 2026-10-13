from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef

from funcs import *
from preprocessing import data_loaders
from models import build_model

MODEL_CHECKPOINTS = {
    "small_cnn": CHECKPOINT_DIR / "small_cnn_best.pth",
    "resnet18": CHECKPOINT_DIR / "resnet18_best.pth",
    "vit_b_16": CHECKPOINT_DIR / "vit_b_16_best.pth",
    "efficientnet_b0": CHECKPOINT_DIR / "efficientnet_b0_best.pth",
    "alexnet": CHECKPOINT_DIR / "alexnet_best.pth",
    "efficientnet_b0_scratch": CHECKPOINT_DIR / "efficientnet_b0_scratch_best.pth",
    "efficientnet_b0_finetuned": CHECKPOINT_DIR / "efficientnet_b0_finetuned_best.pth",
}

def macro_specificity(y_true, y_pred, num_classes):
    """Mean one-vs-rest specificity across all classes."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    total = cm.sum()
    per_class = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        per_class.append(float(tn / denom) if denom else 0.0)

    return float(np.mean(per_class))


def get_num_classes_and_names(test_dataset, checkpoint):
    mapping = checkpoint.get("class_to_idx")
    if isinstance(mapping, dict) and mapping:
        mapping = {str(k): int(v) for k, v in mapping.items()}
        names = [name for name, idx in sorted(mapping.items(), key=lambda x: x[1])
        ]
        return len(names), names, mapping

    df = test_dataset.df
    
    if "class_name" in df.columns:
        pairs = df[["class_name", "label"]].drop_duplicates()
        mapping = {str(row["class_name"]): int(row["label"]) for _, row in pairs.iterrows()}
        names = [name for name, idx in sorted(mapping.items(), key=lambda x: x[1])]
        labels = sorted(df["label"].astype(int).unique().tolist())
        return len(names), names, mapping
    
    labels = sorted(df["label"].astype(int).unique().tolist())
    
    return len(labels), [str(i) for i in labels], None


# -------------------------------- Main --------------------------------

def main():
    set_env()
    data = data_loaders()
    _, _, test_dataset = data["datasets"]
    _, _, test_loader = data["loaders"]

    test_df = test_dataset.df
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test label counts:\n{test_df['label'].value_counts().sort_index()}")
    print(f"Evaluation device: {DEVICE}")

    reference = None
    for checkpoint_path in MODEL_CHECKPOINTS.values():
        if checkpoint_path.exists():
            reference = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            break

    num_classes, class_names, class_to_idx = get_num_classes_and_names(test_dataset, reference)
    print(f"Classes ({num_classes}): {class_names}")

    summary_rows = []

    for model_name, checkpoint_path in MODEL_CHECKPOINTS.items():
        print(f"\n{'=' * 60}\nEvaluating {model_name}")

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))

        model = build_model(model_name, num_classes=num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        model.eval()

        y_true, y_pred, y_prob = [], [], []
        inference_start = time.perf_counter()

        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                logits = model(images)

                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                probabilities = torch.softmax(logits, dim=1)
                predictions = probabilities.argmax(dim=1)

                y_true.extend(labels.numpy().astype(int).tolist())
                y_pred.extend(predictions.cpu().numpy().astype(int).tolist())
                y_prob.extend(probabilities.cpu().numpy().tolist())

        inference_seconds = time.perf_counter() - inference_start

        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        label_ids = list(range(num_classes))

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
        specificity = macro_specificity(y_true, y_pred, num_classes)
        mcc = matthews_corrcoef(y_true, y_pred)

        report = classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=label_ids)

        pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / f"{model_name}_classification_report.csv")

        # Confusion matrix
        pd.DataFrame(
            cm,
            index=[f"true_{name}" for name in class_names],
            columns=[f"pred_{name}" for name in class_names],
        ).to_csv(OUTPUT_DIR / f"{model_name}_confusion_matrix.csv")

        prediction_data = {
            "true_label": y_true,
            "predicted_label": y_pred,
            "true_class": [class_names[i] for i in y_true],
            "predicted_class": [class_names[i] for i in y_pred],
            "correct": y_true == y_pred,
        }
        for i, name in enumerate(class_names):
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
            prediction_data[f"prob_{safe_name}"] = y_prob[:, i]

        if "path" in test_df.columns:
            prediction_data["path"] = test_df["path"].astype(str).tolist()

        pd.DataFrame(prediction_data).to_csv(OUTPUT_DIR / f"{model_name}_predictions.csv", index=False)

        best_epoch = checkpoint.get("epoch")
        validation_metrics = checkpoint.get("validation_metrics", {})
        val_macro_f1 = (validation_metrics.get("macro_f1") if isinstance(validation_metrics, dict) else None)

        row = {
            "model": model_name,
            "test_samples": len(y_true),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "macro_specificity": float(specificity),
            "mcc": float(mcc),
            "inference_seconds": float(inference_seconds),
            "ms_per_image": float(1000 * inference_seconds / len(y_true)),
            "checkpoint_epoch": best_epoch,
            "validation_macro_f1": val_macro_f1,
        }
        
        summary_rows.append(row)

        print(
            f"Accuracy:           {accuracy:.4f}\n"
            f"Macro-F1:           {macro_f1:.4f}\n"
            f"Macro-specificity:  {specificity:.4f}\n"
            f"MCC:                {mcc:.4f}\n"
            f"Inference time:     {inference_seconds:.2f} s "
            f"({row['ms_per_image']:.2f} ms/image)"
        )

        del model

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary.to_csv(OUTPUT_DIR / "test_metrics_summary.csv", index=False)

    print(f"\n{'=' * 60}\nFINAL TEST SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nAll evaluation outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
