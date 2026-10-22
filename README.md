# 🌿 PlantGuard AI – Plant Disease Detection Using Deep Learning

**Course:** CSM422 – Deep Learning

PlantGuard AI looks at a photo of a plant leaf and predicts whether the plant is **healthy**
or which **disease** it has. It then shows the **treatment and prevention tips** and a
**Grad-CAM heatmap** that highlights the part of the leaf the AI used for its decision.

It uses **MobileNetV2 transfer learning** with a small softmax classification head, implemented with Keras/TensorFlow. The model predicts only the six classes listed in `model/class_names.json`; it is not a general-purpose plant diagnostician.

## Features
- MobileNetV2 feature extractor pre-trained on ImageNet, frozen while training the classifier head
- Automatic dataset download (PlantVillage) or a 2-minute demo dataset
- Data augmentation, early stopping, learning-rate reduction
- Web app (Streamlit) with upload or camera, confidence, treatment advice, Grad-CAM
- Command-line prediction
- Project report and viva Q&A in the `docs/` folder

## Project structure
```
PlantGuardAI/
├── dataset/              images, one sub-folder per class
├── model/                trained model + class names (created by train.py)
├── outputs/              graphs and reports (created by train.py)
├── docs/                 PROJECT_REPORT.md, VIVA_QA.md
├── config.py             settings (image size, epochs, paths)
├── download_data.py      downloads real PlantVillage images
├── make_demo_data.py     makes a small fake dataset for a quick test
├── train.py              builds, trains and evaluates the CNN
├── predict.py            predict one image (terminal)
├── app.py                web app (Streamlit)
├── gradcam.py            heatmap code
├── disease_info.py       description + treatment for each disease
├── utils.py              helper functions
└── requirements.txt
```

## Setup in VS Code
1. Install **Python 3.10, 3.11 or 3.12** (TensorFlow does not support very new Python versions).
2. Unzip the project, open the folder in VS Code (`File → Open Folder`).
3. Open a terminal (`Terminal → New Terminal`) and run:
   ```
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   Select the `.venv` interpreter in VS Code (`Ctrl+Shift+P → Python: Select Interpreter`).

## Step 1 – Get the data (choose ONE option)

**Option A – Real data, automatic (recommended):**
```
python download_data.py
```
Downloads about 800 MB once and saves 6 classes (Tomato and Potato: healthy, early blight,
late blight) x 400 images. More images: `python download_data.py --per_class 800`.
See all class names: `python download_data.py --list`.

**Option B – Real data, manual (if Option A fails):**
Download the *PlantVillage* dataset from Kaggle, take the **color** images and copy 4–8 class
folders into `dataset/` like this:
```
dataset/
  Tomato___healthy/        img1.jpg ...
  Tomato___Early_blight/   ...
```

**Option C – Quick test (no download):**
```
python make_demo_data.py
```
Creates 3 synthetic classes. It is only to check that the project runs; use real data
for your final results.

> Only keep the classes you want inside `dataset/`. Delete demo folders before using real data.

## Step 2 – Train
```
python train.py
```
- Takes a few minutes on a normal laptop (CPU) for roughly 2,152 images in the included dataset (actual time depends on hardware and TensorFlow configuration).
- Options: `--epochs 20`, `--img_size 96` (faster), `--batch 16` (less memory).
- Creates `model/plantguard_model.keras` and evaluation files in `outputs/`. The included metrics describe a previous run; retraining may produce different results.

## Step 3 – Use it
Web app:
```
streamlit run app.py
```
Terminal:
```
python predict.py path/to/leaf.jpg
```

## How the model works
| Stage | Purpose |
|-------|---------|
| Input 128×128×3 | Resized RGB leaf image |
| RandomFlip / Rotation / Zoom | Small training-time image variations |
| Rescaling to −1…1 | Matches MobileNetV2 input preprocessing |
| MobileNetV2 (`weights="imagenet"`, `include_top=False`) | Reuses general visual features learned from ImageNet; base is frozen during this training run |
| GlobalAveragePooling2D | Compresses spatial feature maps into a feature vector |
| Dropout(0.25) | Regularises the classifier head during training |
| Dense(softmax) | Produces a score for each known class |

Loss: sparse categorical cross-entropy · Optimizer: Adam. The saved model should be evaluated on a separate, representative test set before being used for real decisions.

## Syllabus mapping (CSM422)
| Unit | Topic | Where in the project |
|------|-------|----------------------|
| I | Activation functions (ReLU, softmax), parameters/hyperparameters | `train.py`, `config.py` |
| II | Information flow and dimensions for image inputs, feedforward | `model.summary()` output while training |
| III | Loss, backpropagation, gradient descent, training in batches | `model.fit()` (Adam, batch size 32) |
| IV | Convolution, stride, padding | Convolutional layers are part of the pretrained MobileNetV2 backbone; inspect `model.summary()` during training for the layer names and output shapes. This project does not define a separate custom `Conv2D` layer. |
| V | Feature maps, pooling, image classification | `train.py` (MobileNetV2 feature maps and `GlobalAveragePooling2D`) and `gradcam.py` |

## Troubleshooting
| Problem | Fix |
|---------|-----|
| `No module named tensorflow` | Use Python 3.10–3.12 and run `pip install -r requirements.txt` inside the venv |
| `Found 0 files` / no images | Check `dataset/` has sub-folders that contain images |
| Training too slow | `python train.py --img_size 96 --epochs 8` or use fewer classes |
| Out of memory | `python train.py --batch 16` |
| App says model not found | Run `python train.py` first |
| `download_data.py` fails | Use Option B (manual Kaggle download) |


## Reproducibility and honest reporting
- `train.py` uses a deterministic random seed and a per-class image cap. It makes an 80/20 train/validation split **within each class**.
- The validation score is not a guarantee of real-world accuracy. Images from the same source or near-duplicates can make validation results optimistic; a separate test set is preferable.
- Training downloads MobileNetV2 ImageNet weights on the first run, so an internet connection is required. The included `.keras` file is the already-saved model; training again will replace it and regenerate files in `outputs/`.
- Keep the report's model description, dataset counts, and metrics aligned with the actual files generated by your run. Do not claim experiments or results you did not perform.
