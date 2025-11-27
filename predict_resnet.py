import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# 1. CONFIG
MODEL_PATH = "deepweeds_resnet_attention(1).keras"
IMG_SIZE = 224
class_names = [
    "Chinee apple",      # 0
    "Lantana",           # 1
    "Parkinsonia",       # 2
    "Parthenium",        # 3
    "Prickly acacia",    # 4
    "Rubber vine",       # 5
    "Siam weed",         # 6
    "Snake weed",        # 7
    "Negative",          # 8
]  # Update with actual DeepWeeds class names

# 2. DEFINE CUSTOM LAYER (squeeze_excite_block)
def squeeze_excite_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation block for channel attention"""
    channels = input_tensor.shape[-1]
    
    # Squeeze: Global Average Pooling
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    
    # Excite: Two dense layers
    excitation = tf.keras.layers.Dense(channels // ratio, activation='relu')(squeeze)
    excitation = tf.keras.layers.Dense(channels, activation='sigmoid')(excitation)
    
    # Reshape for multiplication
    excitation = tf.keras.layers.Reshape((1, 1, channels))(excitation)
    
    # Scale: Multiply input by excitation weights
    scaled = tf.keras.layers.Multiply()([input_tensor, excitation])
    return scaled

# 3. LOAD MODEL with custom_objects
custom_objects = {"squeeze_excite_block": squeeze_excite_block}
model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

def load_and_preprocess(img_path):
    """Load image from disk and preprocess like during training."""
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)          # shape (H, W, 3), 0–255
    x = x / 255.0                        # match ImageDataGenerator(rescale=1./255)
    x = np.expand_dims(x, axis=0)        # shape (1, H, W, 3)
    return x

def predict_image(img_path):
    x = load_and_preprocess(img_path)
    preds = model.predict(x)
    pred_idx = np.argmax(preds[0])
    pred_prob = float(np.max(preds[0]))
    pred_class = class_names[pred_idx]
    return pred_class, pred_prob, pred_idx

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    pred_class, pred_prob, pred_idx = predict_image(img_path)
    print(f"Image: {img_path}")
    print(f"Predicted class index: {pred_idx}")
    print(f"Predicted class name:  {pred_class}")
    print(f"Confidence:            {pred_prob:.4f}")
