# PlantGuard AI — Viva Preparation
Use these as study notes. Answer in your own words and be honest about which parts you implemented, changed, and tested.

**1. What does the project do?**
It classifies a leaf image into one of six potato/tomato categories supported by the saved model, then shows the top scores, general guidance, and an optional Grad-CAM overlay.

**2. Which model does the code use?**
MobileNetV2 pretrained on ImageNet as a frozen feature extractor, followed by global average pooling, dropout, and a dense softmax classifier.

**3. What is transfer learning?**
Using features learned on a large dataset for a related task, rather than learning every visual feature from scratch.

**4. Why MobileNetV2?**
It is designed to be relatively efficient and provides useful pretrained visual features. This project freezes its base to keep the training workload smaller.

**5. What does `include_top=False` mean?**
It omits MobileNetV2's original ImageNet classification head so this project can attach its own six-class head.

**6. Why resize images?**
The model expects a consistent input shape, 128×128 RGB by default.

**7. Why scale pixels to −1…1?**
That is the input range used by MobileNetV2's preprocessing convention; using the same preprocessing at training and prediction time matters.

**8. What is data augmentation?**
Applying small transformations such as flips, rotations, and zooms during training to expose the model to varied versions of training images.

**9. What are training and validation sets?**
The training set updates model weights. The validation set monitors performance on held-out examples during training; it is not a substitute for a final independent test set.

**10. Which loss and optimizer are used?**
Sparse categorical cross-entropy for integer class labels and Adam to update trainable parameters.

**11. What is softmax?**
It converts the output logits into class scores that sum to one. These scores are conditional on the classes the model knows.

**12. What is an epoch and batch size?**
An epoch is one pass through the training examples. Batch size is the number of examples processed together before a training update.

**13. What is overfitting?**
When a model fits training examples but performs poorly on new data. Augmentation, dropout, early stopping, and evaluation on genuinely separate data can help assess or reduce it.

**14. What does a confusion matrix show?**
Counts of actual versus predicted labels; off-diagonal values show which categories are being confused.

**15. What are precision, recall, and F1-score?**
Precision measures how often predictions for a class are correct; recall measures how many true examples of that class are found; F1 is their harmonic mean.

**16. What is Grad-CAM?**
A gradient-based technique that produces a heatmap of image regions influencing a class score. It is approximate and does not prove the model's reasoning is correct.

**17. What does 90.95% validation accuracy mean here?**
That is the value recorded in the included metrics file for one historical validation run. It is not a guarantee of accuracy on field images or an independent test result.

**18. Why can validation results be optimistic?**
Similar or duplicate images may appear in both training and validation sets. A duplicate-aware split and an independent test set give stronger evidence.

**19. What are the project's limitations?**
It only predicts six known classes, may struggle with field lighting/backgrounds or unknown diseases, and its advice is general rather than expert diagnosis.

**20. What would you improve next?**
Evaluate on a source-documented independent set of field images, check duplicates, report per-class metrics, and add out-of-scope detection.

**21. How do you run the app?**
Activate the project's virtual environment, install `requirements.txt`, then run `python -m streamlit run app.py`.

**22. Which files should you explain in a viva?**
`train.py` for the model and evaluation, `predict.py`/`utils.py` for preprocessing and inference, `app.py` for the interface, and `gradcam.py` for the heatmap.
