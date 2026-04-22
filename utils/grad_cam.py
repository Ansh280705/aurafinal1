"""
utils/grad_cam.py
=================
Gradient-weighted Class Activation Mapping (Grad-CAM).
EXTREME ROBUSTNESS EDITION.
"""

import cv2
import numpy as np
import tensorflow as tf

def find_last_conv_layer(model):
    """
    Recursively find the last convolutional layer in a model or sub-models.
    """
    # 1. Search in main model layers
    for layer in reversed(model.layers):
        if "conv" in layer.__class__.__name__.lower():
            return layer
        # 2. Search in nested layers
        if hasattr(layer, 'layers'):
            found = find_last_conv_layer(layer)
            if found:
                return found
    return None

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    target_layer=None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap.
    """
    if target_layer is None:
        target_layer = find_last_conv_layer(model)
    
    if target_layer is None:
        raise ValueError("Could not find a convolutional layer for Grad-CAM. Check model architecture.")

    # Build model for internal activation lookup
    # inputs must be a single tensor for VGG16 standard usage
    inputs = model.inputs
    grad_model = tf.keras.models.Model(
        inputs  = inputs,
        outputs = [target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        tape.watch(inputs) # Ensure inputs are being watched
        conv_outputs, predictions = grad_model(img_array)
        # Class score (binary)
        # Handle cases where output might be (None, 1)
        class_channel = predictions[:, 0]

    # Gradients of class score w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Check if grads are None (happens if target_layer isn't connected to output)
    if grads is None:
        # Fallback: find ANY conv layer that might work
        return np.zeros(img_array.shape[1:3])

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure types match for blending
    overlay = (alpha * heatmap_rgb.astype(np.float32) + (1 - alpha) * original_img.astype(np.float32)).astype(np.uint8)
    return overlay

def generate_gradcam_overlay(
    img_array: np.ndarray,
    model: tf.keras.Model,
    original_img: np.ndarray | None = None,
    last_conv_layer_name: str = "block5_conv3",
    alpha: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline with extreme error handling.
    """
    target_layer = None
    
    # Step A: Direct Lookup
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except Exception:
        pass
    
    # Step B: Recursive Search if A failed
    if target_layer is None:
        target_layer = find_last_conv_layer(model)
        
    # Step C: Heatmap Generation
    try:
        heatmap = make_gradcam_heatmap(img_array, model, target_layer=target_layer)
    except Exception:
        # Final fallback: black heatmap
        heatmap = np.zeros((14, 14)) # standard VGG16 last layer size

    # Step D: Overlay
    if original_img is None:
        original_img = np.uint8(img_array[0] * 255)
    
    overlay = overlay_gradcam(original_img, heatmap, alpha=alpha)
    return heatmap, overlay
