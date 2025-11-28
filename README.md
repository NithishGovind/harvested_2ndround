# Weed Detection and Classification Project

## 1. Project Story and Implementation Plan

The goal of this project is to build a robust weed detection system for agricultural fields using deep learning and computer vision. The work is organized into three stages, two completed (classification and detection) and one planned.

### 1.1 Weed Classification (Completed)

The project initially focused on weed species classification using field images. A baseline ResNet‑50 model was trained on the DeepWeeds dataset (17,509 images, 9 classes), providing a strong starting point for image-level weed recognition.

A second model, ResNet‑50 + Attention, was then developed to improve performance and robustness under real-world conditions.

### 1.2 Weed Detection with YOLO (Current Focus)

After classification, the project moved to spatial localization of weeds using object detection. A YOLO-based detector was trained on a crop–weed bounding-box dataset to detect and separate crops from weeds at the object level. The detector achieves high mAP, precision, and recall and is suitable for deployment on GPU or edge devices.

This detection stage focuses on:

- Crop vs Weed object detection
- Evaluation with mAP@0.5, mAP@0.5:0.95, IoU, Precision, Recall, F1‑score
- Confusion matrix analysis
- Inference latency and FPS on GPU/CPU
- Failure-case visualization (8+ images under different camera/lighting conditions)


## 2. Methodology Overview

- Use the DeepWeeds dataset for species-level classification and benchmarking of ResNet‑50 and ResNet‑50 + Attention.
- Use a crop–weed detection dataset (Weeds Detection) for training YOLO-based detectors at object level.
- Train baseline and attention-augmented ResNet‑50 models; compare using accuracy, macro/weighted precision, recall, F1, mAP, and IoU.
- Train a YOLO detector for Crop vs Weed detection and evaluate using mAP@0.5, mAP@0.5:0.95, IoU, Precision, Recall, F1‑score, confusion matrix, and FPS.
- Perform robustness analysis under camera and lighting distortions (brightness, color temperature shifts, light motion blur) and collect failure cases.
- Use these findings to guide real-world deployment recommendations (camera settings, lighting constraints, threshold tuning, model choice for field hardware).

---

## 3. Model Comparison – ResNet‑50 vs ResNet‑50 + Attention (Classification)

Evaluation on the DeepWeeds test set (1,752 images):

| Metric              | ResNet‑50 Baseline | ResNet‑50 + Attention | Improvement |
|---------------------|-------------------|------------------------|------------|
| Training Epochs     | 77                | 55                     | −22 epochs |
| Test Accuracy       | 77.0%             | 81.0%                  | +4.0%      |
| Macro Precision     | 71.9%             | 74.1%                  | +2.2%      |
| Macro Recall        | 78.7%             | 85.3%                  | +6.6%      |
| Macro F1‑Score      | 73.1%             | 78.4%                  | +5.3%      |
| Weighted F1         | 78.0%             | 82.0%                  | +4.0%      |
| mAP                 | 85.5%             | 89.0%                  | +3.5%      |
| Mean IoU            | 58.5%             | 65.2%                  | +6.7%      |

**Key Takeaways**

- The attention mechanism provides consistent improvements across all metrics.
- Faster convergence: the attention model reaches superior performance in 55 epochs vs 77 for the baseline (≈29% fewer epochs).
- Largest gains occur in macro recall and mean IoU, indicating better detection of minority weed classes.
- The enhanced model achieves 81% accuracy with stronger generalization to underrepresented species.
- Recommended classifier for deployment: **ResNet‑50 + Attention**.

Detailed classification reports:

- Baseline ResNet‑50 classification report: `Resnet_classification.md`  
- ResNet‑50 + Attention classification report: `resnet_attention_classification.md`

---

## 4. YOLO Weed Detection – Data, Model, and Training

### 4.1 Detection Dataset and Preparation

- Dataset: Crop–weed detection dataset with YOLO-format labels.
- Classes:
  - `0` – Crop
  - `1` – Weed
- Splits:
  - Train: images + labels
  - Validation: images + labels
  - Test: images + labels (used for final evaluation / qualitative analysis)
- Input size: 640×640 (images are resized/letterboxed by YOLO).
- Labels: YOLO format – `class_id cx cy w h` normalized to [0, 1].

Data augmentation (example configuration):

- Mosaic combination of 4 images
- Horizontal flip (50% probability)
- HSV color augmentation (hue, saturation, value jitter)
- Random translate and scale
- Random erasing on patches

These augmentations help the detector generalize to varying camera viewpoints and lighting conditions.

### 4.2 YOLO Model Architecture and Hyperparameters

- Detector: YOLO nano model (Ultralytics YOLOv8/YOLOv11-style)
- Parameters: ≈3M
- FLOPs: ≈8 GFLOPs at 640×640
- Core hyperparameters (typical run):

  - `epochs`: 50  
  - `batch`: 16  
  - `imgsz`: 640  
  - `optimizer`: auto (SGD by default)  
  - `lr0`: 0.01 (initial LR)  
  - `lrf`: 0.01 (final LR ratio)  
  - `momentum`: 0.937  
  - `weight_decay`: 0.0005  
  - `iou`: 0.7 (NMS IoU threshold)  
  - `max_det`: 300  

Loss weights:

- `box`: 7.5 – bounding box regression loss (IoU-based)
- `cls`: 0.5 – classification loss (Crop vs Weed)
- `dfl`: 1.5 – Distribution Focal Loss for precise box edges

Total loss per batch is a weighted sum of these components.

### 4.3 Training Environment and Duration

- Hardware:
  - GPU: Tesla P100 (16 GB VRAM)
  - Framework: Ultralytics YOLO (PyTorch backend)
  - Python 3.10
- Training:
  - 50 epochs
  - ≈0.22 hours (~13 minutes) total on GPU
  - ~5.7 iterations/second, ≈15–16 seconds per epoch
- Training artifacts:
  - `runs/detect/train/weights/best.pt` – best-performing checkpoint
  - `runs/detect/train/results.csv` – per-epoch metrics
  - `runs/detect/train/results.png` – loss and mAP curves
  - Curves: `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png`
  - Confusion matrices: `confusion_matrix.png`, `confusion_matrix_normalized.png`

---

## 5. YOLO Performance Evaluation Report

### 5.1 Detection Metrics

(Values below follow the structure you provided; adjust if you re-train.)

**Validation summary (all classes):**

- Precision (P): ≈ 0.89  
- Recall (R): ≈ 0.85  
- mAP@0.5: ≈ 0.92  
- mAP@0.5:0.95: ≈ 0.66  

**Per-class example:**

- Crop:
  - Precision: ≈ 0.84  
  - Recall: ≈ 0.81  
  - mAP@0.5: ≈ 0.90  
  - mAP@0.5:0.95: ≈ 0.67  

- Weed:
  - Precision: ≈ 0.95  
  - Recall: ≈ 0.89  
  - mAP@0.5: ≈ 0.94  
  - mAP@0.5:0.95: ≈ 0.65  

**Interpretation:**

- High **mAP@0.5** indicates the detector reliably finds crops and weeds with sufficient overlap (IoU ≥ 0.5).
- **Weed detection** is particularly strong (high precision and recall), which is crucial for minimizing missed weeds and false alarms on crops.
- **mAP@0.5:0.95** shows that boxes remain reasonably tight even under stricter IoU thresholds, which is important for precise localization.

### 5.2 IoU, Precision, Recall, and F1

- IoU (Intersection over Union) is used to decide when a predicted box matches a ground-truth box.
- For mAP@0.5, a detection is correct if IoU ≥ 0.5 and the class matches.
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

In this project, F1 for detection is around 0.87, indicating a good balance between not missing weeds (recall) and not over-detecting (precision).

### 5.3 Confusion Matrix

Ultralytics generates confusion matrices for the detector:

- `/detect/trainconfusion_matrix.png` – raw counts
- `/detect/trainconfusion_matrix_normalized.png` – normalized by ground truth frequencies

They show:

- Strong diagonal entries (Crop→Crop, Weed→Weed).
- Weak off-diagonal entries, meaning few misclassifications between crop and weed classes.

These figures are included in the report to visually summarize detector performance.

### 5.4 Inference Latency and FPS

From validation

## Detailed Model Reports


For detailed evaluation, metrics, visualizations, and robustness analysis of each model, see:


- **Baseline ResNet-50 classification report:**  
  [ResNet-50 Evaluation](Resnet_classification.md)


- **ResNet-50 + Attention classification report:**  
  [ResNet-50 + Attention Evaluation](resnet_attention_classification.md)

- **YOLO Detection report:**
  [Yolo Evaluation](yolo_detection.md)