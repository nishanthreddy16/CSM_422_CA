# PlantGuard AI — Plant Leaf Disease Classification
**Course:** CSM422 – Deep Learning | **Session:** 2026–27

## 1. Abstract
PlantGuard AI is an educational image-classification application for six potato and tomato leaf categories. It uses a MobileNetV2 feature extractor with a small classification head, displays the top predictions, provides general plant-care guidance, and attempts to show a Grad-CAM heatmap. It is a prototype, not a confirmed agricultural diagnosis tool.

## 2. Problem statement and objectives
Leaf symptoms can be difficult for a non-specialist to recognise. This project explores whether image classification can help organise a leaf photograph into one of the categories represented in its training data. Objectives are to prepare labelled images, train and evaluate a model, build a Streamlit interface, and present an approximate visual explanation.

## 3. Dataset and labels
The code expects one folder per class inside `dataset/`. The six configured label names in the supplied model are:
- Potato — Early blight
- Potato — Late blight
- Potato — Healthy
- Tomato — Early blight
- Tomato — Late blight
- Tomato — Healthy

The project includes a subset of public PlantVillage-style images. Check the dataset source and its licence/terms before redistributing the images. `train.py` shuffles each class with a fixed seed, caps the number of images per class, and uses an 80/20 split for training and validation. The actual counts for a new training run are printed to the terminal and saved in `outputs/metrics.json`; report those values rather than assuming all classes have equal counts.

## 4. Preprocessing and augmentation
Images are decoded as RGB and resized to 128 × 128 pixels by default. Random horizontal flips, small rotations, and zooms are applied in the model pipeline during training. Pixel values are rescaled to the −1 to 1 range expected by MobileNetV2.

## 5. Model architecture
The implementation in `train.py` uses:
1. Input layer for an RGB image.
2. RandomFlip, RandomRotation, and RandomZoom augmentation layers.
3. Rescaling to the MobileNetV2 input range.
4. MobileNetV2 with ImageNet weights and `include_top=False`; the pretrained base is frozen during the supplied training procedure.
5. GlobalAveragePooling2D and Dropout(0.25).
6. Dense softmax output with one score per class.

The loss is sparse categorical cross-entropy and the optimizer is Adam. Early stopping and learning-rate reduction callbacks are used. This is transfer learning, not the four-block custom CNN described in some earlier draft material.

## 6. Application workflow
The Streamlit app loads the saved Keras model and class names, accepts an uploaded image or camera photo, resizes it using the shared preprocessing helper, and displays the top three class scores. It shows general guidance from `disease_info.py` and attempts a Grad-CAM overlay. A high score is the model's confidence among these six classes; it does not establish that the image belongs to one of them or that the treatment advice is correct.

## 7. Evaluation and results
The included `outputs/metrics.json` records a historical validation accuracy of **90.95%**, six classes, and five epochs for the run represented by that file. Its recorded per-class image counts are:
- Potato Early blight: 400
- Potato Late blight: 400
- Potato healthy: 152
- Tomato Early blight: 400
- Tomato Late blight: 400
- Tomato healthy: 400

Treat this as a recorded validation result, not an independent test result or a guarantee of field performance. Re-running `train.py` may change the metrics and overwrite the output files. Include the generated confusion matrix, classification report, and training curves when presenting results, and state clearly which run they came from.

## 8. Limitations and responsible use
- Only the six labels supported by the model can be predicted; an unknown plant or disease may still receive a misleading class score.
- Dataset-style images may have cleaner backgrounds than real field photographs.
- Validation accuracy can be optimistic if related or near-duplicate images occur across the split.
- Grad-CAM is an approximate visual explanation and does not prove the model used biologically meaningful evidence.
- Treatment information is general guidance, not a diagnosis. Confirm serious crop problems with an agricultural extension officer or qualified expert.

## 9. Future improvements
Create a separate, source-documented test set of field photographs; test for duplicate images before splitting; report per-class precision, recall, and F1; add an explicit unknown/out-of-scope warning; and validate advice with reliable agricultural sources.

## 10. References
1. Sandler et al., “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” CVPR 2018.
2. Hughes and Salathé, “An open access repository of images on plant health,” PlantVillage, 2015.
3. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” ICCV 2017.
4. TensorFlow/Keras documentation for MobileNetV2 and model evaluation.
