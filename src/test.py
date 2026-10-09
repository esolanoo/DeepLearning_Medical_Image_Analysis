import time
import torch.nn as nn
from preprocessing import data_loaders
from torch.utils.data import DataLoader
from models import build_model
from funcs import *




data = data_loaders()
train_loader, val_loader, test_loader = [loader for loader in data['loaders']]
train_ds, val_ds, test_ds = [ds for ds in data['datasets']]


print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ), "GB")
else:
    print("Running on CPU")

print("Dataset size:", len(train_ds))
print("Starting DataLoader test...")

start = time.perf_counter()

# Test with zero workers first
test_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

images, labels = next(iter(test_loader))

print("First batch loaded in:",
      round(time.perf_counter() - start, 2), "seconds")
print("Image shape:", images.shape)
print("Labels shape:", labels.shape)

model = build_model("resnet18").to(DEVICE)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,
)
criterion = nn.CrossEntropyLoss()

images, labels = next(iter(test_loader))
images = images.to(DEVICE)
labels = labels.to(DEVICE)

start = time.perf_counter()

logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

print("One training step:", round(time.perf_counter() - start, 2), "seconds")