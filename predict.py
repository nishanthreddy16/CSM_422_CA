"""
Predict the disease of ONE leaf image (command line).

Run:  python predict.py path/to/leaf.jpg
"""
import sys

import numpy as np
from PIL import Image

import config
import utils
from disease_info import GENERAL_TIPS, get_info
from gradcam import make_gradcam, overlay_heatmap

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

model = utils.load_model()
class_names = utils.load_class_names()
img_size = model.input_shape[1]

pil_img = Image.open(sys.argv[1]).convert("RGB")
batch = utils.prepare_image(pil_img, img_size)
probs = model.predict(batch, verbose=0)[0]
top = np.argsort(probs)[::-1][:3]
best = class_names[top[0]]
info = get_info(best)

print("\n==============  PlantGuard AI  ==============")
print("Prediction :", utils.pretty_name(best))
print(f"Confidence : {probs[top[0]] * 100:.2f}%")
print("Status     :", info["status"])
if probs[top[0]] < config.CONFIDENCE_THRESHOLD:
    print("WARNING    : Low confidence - try a clearer, close-up photo of a single leaf.")
print("\nAbout      :", info["description"])
print("Guidance   :", info["treatment"], "(general guidance only)")
print("Prevention :", info["prevention"])
print("\nTop 3 predictions:")
for i in top:
    print(f"  {utils.pretty_name(class_names[i]):45s} {probs[i] * 100:6.2f}%")
print("\nNote:", GENERAL_TIPS)

# Grad-CAM heatmap (saved to outputs/)
try:
    cam, _ = make_gradcam(model, batch)
    result = overlay_heatmap(pil_img.resize((img_size * 2, img_size * 2)), cam)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "gradcam_result.png"
    result.save(out)
    print(f"\nGrad-CAM heatmap saved to {out}")
except Exception as e:
    print("\n(Grad-CAM skipped:", e, ")")
