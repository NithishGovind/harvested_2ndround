import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# 1. CONFIG
MODEL_PATH = "deepweeds_resnet_attention.keras"   # your .keras file
IMG_SIZE = 224                                    # set to your training size
class_names = [
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
    "class_7",
    "class_8",
]  # replace with your actual class names in correct order

# If you used a custom layer (e.g., squeeze_excite_block), import/define it here
# from your_module import squeeze_excite_block


# 2. LOAD MODEL
model = keras.models.load_model(MODEL_PATH)

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
