"""
Grad-CAM: shows WHICH part of the leaf the CNN looked at to make its decision.
Red/yellow = areas that influenced the prediction the most.
"""
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


def make_gradcam(model, img_batch, layer_name="last_conv"):
    """Returns (heatmap 0-1, predicted class index)."""
    grad_model = keras.Model(model.inputs,
                             [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.convert_to_tensor(img_batch))
        class_idx = tf.argmax(preds[0])
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), int(class_idx)


def overlay_heatmap(pil_image, cam, alpha=0.45):
    """Blend the heatmap on top of the original image and return a PIL image."""
    pil_image = pil_image.convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(pil_image.size, Image.BILINEAR)
    heat = matplotlib.colormaps["jet"](np.array(cam_img) / 255.0)[..., :3]
    base = np.array(pil_image, dtype="float32") / 255.0
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))
