# Submission Checklist

1. [ ] Delete demo data from `dataset/`, then run `python download_data.py` (real PlantVillage data).
2. [ ] Run `python train.py` and wait until it finishes.
3. [ ] Copy YOUR real results into `docs/PROJECT_REPORT.md` section 8:
   - Validation accuracy (printed at the end of training and in `outputs/metrics.json`)
   - Precision, recall, F1 (`outputs/classification_report.txt`)
   - Confusion matrix (`outputs/confusion_matrix.png`)
   - Training/validation accuracy and loss graphs (`outputs/training_curves.png`)
   Do not write numbers you did not get from your own run.
4. [ ] Run `streamlit run app.py` and test 8-10 different leaf images (healthy and diseased,
       from the validation data and from outside the dataset). Check prediction, confidence,
       guidance text and Grad-CAM heatmap. Note any wrong predictions honestly in the report.
5. [ ] Take screenshots of the app for the report/presentation.
6. [ ] Remember: treatment text is general guidance, not a confirmed diagnosis.

## Presentation tip
Demo flow: upload a diseased leaf -> show prediction + confidence -> show Grad-CAM
("the red area is what the CNN used to decide") -> show confusion matrix and accuracy curves.
