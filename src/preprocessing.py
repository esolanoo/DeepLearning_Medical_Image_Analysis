import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from funcs import *
import warnings

set_env()
warnings.filterwarnings('ignore')


class KvasirDataset(Dataset): # Inherits from torch Datasets
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(row["label"])


def data_loaders():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    cuda = torch.cuda.is_available()
    records = []
    
    base_dataset = ImageFolder(root=DATA_DIR)

    for path, label in base_dataset.samples:
        # No try: since we have checked for errors already
        with Image.open(path) as img:
            img.verify()
        records.append({"path": str(path), "label": label, "class_name": base_dataset.classes[label]})

    df = pd.DataFrame(records)
    
    train_df, temp_df = train_test_split(df, train_size=0.7,  random_state=SEED, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=SEED, stratify=temp_df["label"])

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Transform to use in test and eval so there is no noise introduced
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset = KvasirDataset(train_df, train_transform)
    val_dataset = KvasirDataset(val_df, eval_transform)
    test_dataset = KvasirDataset(test_df, eval_transform)
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=cuda)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=cuda)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=cuda)
    
    return train_loader, val_loader, test_loader