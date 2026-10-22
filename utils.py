"""Small helper functions shared by predict.py and app.py."""
import json

import numpy as np
from PIL import Image

import config


def split_name(class_name):
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight')"""
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
    else:
        plant, disease = "Plant", class_name
    return plant.replace("_", " "), disease.replace("_", " ")


def pretty_name(class_name):
    plant, disease = split_name(class_name)
    return f"{plant} - {disease}"


def load_class_names():
    with open(config.CLASS_NAMES_PATH) as f:
        return json.load(f)


def load_model():
    from tensorflow import keras
    return keras.models.load_model(config.MODEL_PATH)


def prepare_image(pil_image, img_size):
    """PIL image -> (1, H, W, 3) float array. The model rescales 0-255 itself."""
    img = pil_image.convert("RGB").resize((img_size, img_size))
    return np.expand_dims(np.array(img, dtype="float32"), axis=0)
