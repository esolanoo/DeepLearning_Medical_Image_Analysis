import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from funcs import *
import warnings

warnings.filterwarnings('ignore')


class KvasirDataset(Dataset): # Inherits from torch Datasets
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = row['path'].replace("\\", "/")
        with Image.open(IMAGE_DIR / str(p)) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(row["label"])


def data_loaders():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    cuda = torch.cuda.is_available()
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

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
    
    data = {
        'datasets': [train_dataset, val_dataset, test_dataset],
        'loaders': [train_loader, val_loader, test_loader]
    }
    
    return data