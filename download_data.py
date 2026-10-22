"""
Downloads the real PlantVillage dataset automatically (via TensorFlow Datasets)
and saves a balanced subset as image folders inside ./dataset

Run:  python download_data.py                      (6 classes x 400 images, default)
      python download_data.py --per_class 800
      python download_data.py --classes Tomato___healthy Tomato___Early_blight
      python download_data.py --list               (show all 38 class names)

Note: the first run downloads about 800 MB, so you need internet.
If this does not work, download PlantVillage manually from Kaggle (see README).
"""
import argparse

import tensorflow_datasets as tfds
from PIL import Image

import config

DEFAULT_CLASSES = [
    "Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight",
    "Potato___healthy", "Potato___Early_blight", "Potato___Late_blight",
]

parser = argparse.ArgumentParser()
parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
parser.add_argument("--per_class", type=int, default=400)
parser.add_argument("--list", action="store_true", help="only print class names")
args = parser.parse_args()

ds, info = tfds.load("plant_village", split="train", with_info=True)
names = info.features["label"].names

if args.list:
    print("\n".join(names))
    raise SystemExit

wanted = [c for c in args.classes if c in names]
missing = [c for c in args.classes if c not in names]
if missing:
    print("Warning - these classes were not found:", missing)
if not wanted:
    raise SystemExit("No valid classes. Run: python download_data.py --list")

counts = {c: 0 for c in wanted}
for c in wanted:
    (config.DATA_DIR / c).mkdir(parents=True, exist_ok=True)

for example in tfds.as_numpy(ds):
    cname = names[int(example["label"])]
    if cname in counts and counts[cname] < args.per_class:
        Image.fromarray(example["image"]).save(config.DATA_DIR / cname / f"{counts[cname]}.jpg")
        counts[cname] += 1
    if all(v >= args.per_class for v in counts.values()):
        break

print("Saved images per class:")
for c, n in counts.items():
    print(f"  {c}: {n}")
print("Done. Now run: python train.py")
