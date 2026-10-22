"""
Creates a small SYNTHETIC leaf dataset so you can test the whole project in 2 minutes
without downloading anything. For real results use download_data.py (PlantVillage).

Run:  python make_demo_data.py
"""
import random

from PIL import Image, ImageDraw

import config

CLASSES = ["Healthy_leaf", "Brown_spot_disease", "Yellow_blight_disease"]
PER_CLASS = 150
SIZE = 128


def make_leaf(label):
    bg = (random.randint(200, 235),) * 3
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)
    green = (random.randint(20, 60), random.randint(120, 180), random.randint(20, 60))
    if label == "Yellow_blight_disease":
        green = (random.randint(180, 220), random.randint(170, 210), random.randint(20, 50))
    d.ellipse([15, 25, SIZE - 15, SIZE - 25], fill=green)
    d.line([20, SIZE // 2, SIZE - 20, SIZE // 2], fill=(30, 90, 30), width=2)
    if label == "Brown_spot_disease":
        for _ in range(random.randint(6, 14)):
            x, y = random.randint(30, SIZE - 40), random.randint(40, SIZE - 50)
            r = random.randint(3, 8)
            d.ellipse([x, y, x + r, y + r], fill=(110, 60, 20))
    return img


for c in CLASSES:
    folder = config.DATA_DIR / c
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(PER_CLASS):
        make_leaf(c).save(folder / f"{i}.jpg")
print("Demo dataset created in ./dataset with classes:", CLASSES)
