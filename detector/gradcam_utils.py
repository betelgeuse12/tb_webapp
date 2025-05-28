# detector/gradcam_utils.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import base64
import io
from PIL import Image

def generate_grad_cam(model, image_path, size=299, last_conv_layer_name="mixed10"):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Unable to read image.")
        return None

    img = cv2.resize(img, (size, size))
    img_array = np.expand_dims(img, axis=0) / 255.0

    # Grad-CAM logic
    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(last_conv_layer_name).output, model.output])


    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        print("Gradients not computed.")
        return None
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap, (size, size))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(overlay_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()

    return heatmap_base64
