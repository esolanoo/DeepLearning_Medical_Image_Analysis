import torch 
import numpy as np
import random
import os 
from pathlib import Path

global DATA_DIR
global GLOBAL_DIR
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
global LOCAL_DATA

# For local notebooks
path = os.path.abspath(os.getcwd())
if "\\src" in path:
    path = path[:-3]
GLOBAL_DIR  = path
DATA_DIR = r"D:\Documentos\Fotos\data"  
OUTPUT_DIR = GLOBAL_DIR + r"data"
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

# For colab
PROJECT_DIR = Path("/content/DeepLearning_Medical_Image_Analysis")
DATA_DIR = Path("/content/kvasir-data")
IMAGE_DIR = DATA_DIR / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV   = DATA_DIR / "val.csv"
TEST_CSV  = DATA_DIR / "test.csv"
LOCAL_DATA = Path("/content/kvasir-data")


SEED = 5338
IMAGE_SIZE = 224 # Forced by vit_b_16
BATCH_SIZE = 32
NUM_WORKERS = 3 # No point in using mor than 0 if code is run locally. See below issue
                # https://stackoverflow.com/questions/78225920/why-nextitertrain-dataloader-takes-long-execution-time-in-pytorch

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


    