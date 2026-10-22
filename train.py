"""PlantGuard AI training using MobileNetV2 transfer learning.
Run: python train.py [--epochs 5 --per_class 500]
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import config

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=str(config.DATA_DIR))
parser.add_argument('--epochs', type=int, default=config.EPOCHS)
parser.add_argument('--img_size', type=int, default=config.IMG_SIZE)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--per_class', type=int, default=config.MAX_IMAGES_PER_CLASS,
                    help='Maximum images used per class to keep training practical')
args = parser.parse_args()
random.seed(config.SEED)
np.random.seed(config.SEED)
tf.keras.utils.set_random_seed(config.SEED)
config.MODEL_DIR.mkdir(exist_ok=True)
config.OUTPUT_DIR.mkdir(exist_ok=True)

# Build a balanced, capped file list so a huge local dataset does not create thousands of batches.
data_root = Path(args.data)
valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
class_names = sorted(p.name for p in data_root.iterdir() if p.is_dir() and any(
    f.suffix.lower() in valid_ext for f in p.iterdir() if f.is_file()))
if len(class_names) < 2:
    raise SystemExit(f'Expected at least 2 class folders containing images in {data_root}')

train_paths, train_labels, val_paths, val_labels, counts = [], [], [], [], []
for label, name in enumerate(class_names):
    files = sorted(p for p in (data_root / name).iterdir() if p.is_file() and p.suffix.lower() in valid_ext)
    random.shuffle(files)
    files = files[:args.per_class]
    counts.append(len(files))
    if len(files) < 2:
        print(f'Warning: class {name} has only {len(files)} image(s); need more for validation.')
        continue
    cut = max(1, int(len(files) * (1 - config.VALIDATION_SPLIT)))
    cut = min(cut, len(files)-1)
    train_paths.extend(map(str, files[:cut])); train_labels.extend([label] * cut)
    val_paths.extend(map(str, files[cut:])); val_labels.extend([label] * (len(files)-cut))

with open(config.CLASS_NAMES_PATH, 'w') as f:
    json.dump(class_names, f)
print('\nClasses:', class_names)
print('Images per class used:', dict(zip(class_names, counts)))
print(f'Training images: {len(train_paths)} | Validation images: {len(val_paths)}')

IMG = (args.img_size, args.img_size)
def load_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG)
    return image, tf.cast(label, tf.int32)

def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 2000), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)
    return ds
train_ds = make_ds(train_paths, train_labels, True)
val_ds = make_ds(val_paths, val_labels, False)

# Dataset chart
plt.figure(figsize=(9, 4)); plt.bar(class_names, counts)
plt.xticks(rotation=60, ha='right', fontsize=8); plt.ylabel('Images used'); plt.tight_layout()
plt.savefig(config.OUTPUT_DIR / 'class_distribution.png', dpi=120); plt.close()

# Transfer learning: use ImageNet features, train a small classifier head first.
inputs = keras.Input(shape=(*IMG, 3), name='leaf_image')
x = layers.RandomFlip('horizontal')(inputs)
x = layers.RandomRotation(0.08)(x)
x = layers.RandomZoom(0.08)(x)
x = layers.Rescaling(1./127.5, offset=-1, name='mobilenet_scaling')(x)
base = keras.applications.MobileNetV2(input_shape=(*IMG, 3), include_top=False,
                                      weights='imagenet')
base.trainable = False
x = base(x, training=False)
x = layers.Activation('linear', name='last_conv')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(class_names), activation='softmax', name='predictions')(x)
model = keras.Model(inputs, outputs, name='PlantGuard_MobileNetV2')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
model.save(config.MODEL_PATH)
print('Model saved to', config.MODEL_PATH)

# Training curves
plt.figure(figsize=(10,4)); plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train'); plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend(); plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(); plt.tight_layout()
plt.savefig(config.OUTPUT_DIR / 'training_curves.png', dpi=120); plt.close()

# Evaluation
true, pred, all_probs = [], [], []
for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    true.extend(labels.numpy().tolist()); pred.extend(np.argmax(probs, axis=1).tolist()); all_probs.extend(probs.tolist())
acc = float(np.mean(np.array(true) == np.array(pred)))
print(f'\nValidation accuracy: {acc*100:.2f}%\n')
report = classification_report(true, pred, target_names=class_names, labels=range(len(class_names)), zero_division=0)
print(report); (config.OUTPUT_DIR / 'classification_report.txt').write_text(report)
(config.OUTPUT_DIR / 'metrics.json').write_text(json.dumps({'validation_accuracy':acc,'classes':class_names,'epochs_run':len(history.history['loss']),'images_per_class_used':dict(zip(class_names,counts))}, indent=2))
cm = confusion_matrix(true, pred, labels=range(len(class_names)))
plt.figure(figsize=(8,6)); plt.imshow(cm); plt.title('Confusion Matrix'); plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=7); plt.yticks(range(len(class_names)), class_names, fontsize=7)
for i in range(len(class_names)):
    for j in range(len(class_names)): plt.text(j,i,cm[i,j],ha='center',va='center',fontsize=7)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
plt.savefig(config.OUTPUT_DIR / 'confusion_matrix.png', dpi=120); plt.close()
print('Reports saved in outputs/. Start the app with: streamlit run app.py')
