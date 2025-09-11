import os

def get_base_dir():
    if os.path.exists("/content/drive"):
        return "/content/drive/MyDrive/clouds-ml"
    else:
        return os.path.expanduser("~/code/clouds-ml")

BASE_DIR = get_base_dir()
DATA_RAW = f"{BASE_DIR}/data/raw"
DATA_IMAGES = f"{BASE_DIR}/data/images"
DATA_PROCESSED = f"{BASE_DIR}/data/processed"
DATA_INTERIM = f"{BASE_DIR}/data/interim"
MODELS_DIR = f"{BASE_DIR}/models"