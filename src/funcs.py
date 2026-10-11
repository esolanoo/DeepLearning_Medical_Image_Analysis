import torch 
import numpy as np
import random
import os 
from pathlib import Path

global DATA_DIR
global SEED
global IMAGE_SIZE
global BATCH_SIZE
global NUM_WORKERS
global DEVICE
global PROJECT_DIR 
global IMAGE_DIR
global TRAIN_CSV 
global VAL_CSV   
global TEST_CSV 
global MODELS
global PRETRAINED

# Paths
PROJECT_DIR = Path(r"D:\Documentos\UP\MIACD\DeepLearning_Medical_Image_Analysis")
DATA_DIR = Path(r"C:\Users\Eduardo\Datasets")
IMAGE_DIR = DATA_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "data"
TRAIN_CSV = IMAGE_DIR / "train.csv"
VAL_CSV   = IMAGE_DIR / "val.csv"
TEST_CSV  = IMAGE_DIR / "test.csv"
SRC_DIR = PROJECT_DIR / "src"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "results" / "test_evaluation"
RESULT_DIR = PROJECT_DIR / "results"

# Torch
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
MODELS = ["resnet18", "efficientnet_b0", "vit_b_16", "alexnet", "small_cnn"]

# Misc
SEED = 5338
IMAGE_SIZE = 224 # Forced by vit_b_16
BATCH_SIZE = 16
NUM_WORKERS = 0 # No point in using mor than 0 if code is run locally. See below issue
                # https://stackoverflow.com/questions/78225920/why-nextitertrain-dataloader-takes-long-execution-time-in-pytorch
PRETRAINED = True

def set_env():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    set_seed(SEED)
    set_deterministic()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def set_deterministic():
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    