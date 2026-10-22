"""Central settings for PlantGuard AI. Change values here if you want."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "plantguard_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = 128          # images are resized to IMG_SIZE x IMG_SIZE
BATCH_SIZE = 32
EPOCHS = 5
VALIDATION_SPLIT = 0.2  # 80% training, 20% validation
SEED = 42
MAX_IMAGES_PER_CLASS = 500  # caps training at 500 images per class for practical runtime
CONFIDENCE_THRESHOLD = 0.60   # below this the app warns "not sure"
