# MODEL MAYHEM — PCB Defect Detection
**Team**: Zenkai
**Team Members**:
Rajesh Kumar Swain
Abankita Behera
Mohammad Jazbi

## 1. Model Overview

- Architecture: YOLO26m
- Task: Six-class PCB defect object detection
- Input resolution: 896 x 896
- Preprocessing: CLAHE applied to the LAB luminance channel
- Optimizer: AdamW
- Initial learning rate: 0.001
- Maximum training epochs: 120
- Early-stopping patience: 30
- Checkpoint-selection metric: Competition mAP@50:90

The final model was fine-tuned from the official pretrained
`yolo26m.pt` checkpoint.

The previous 640-resolution training checkpoint was unavailable
in the training kernel. Therefore, this run was a fresh
transfer-learning run, not a continuation of the previous run.

## 2. Target Classes

The class IDs must remain in this exact order:

| Class ID | Class Name |
|---|---|
| 0 | missing_hole |
| 1 | mouse_bite |
| 2 | open_circuit |
| 3 | short |
| 4 | spur |
| 5 | spurious_copper |

## 3. Submission Files

- `weights/best.pt` — final trained inference checkpoint
- `code/train.py` — dataset preprocessing and model training
- `code/inference.py` — inference and CSV-generation script
- `predictions.csv` — predictions in the prescribed format
- `requirements.txt` — Python dependencies
- `validation/` — validation metrics and training history
- `test_visualizations/` — test prediction images and contact sheets

## 4. Results
Confusion matrix
<img width="1280" height="960" alt="WhatsApp Image 2026-11-11 at 4 28 12 PM" src="https://github.com/user-attachments/assets/fa303ba4-5b00-4ae6-b7ef-86b0d40e4b07" />

All plots
<img width="1280" height="591" alt="WhatsApp Image 2026-11-11 at 4 27 59 PM" src="https://github.com/user-attachments/assets/82bda09e-3b14-4907-b9c5-4f1ad520f295" />

Testing results
<img width="1280" height="832" alt="WhatsApp Image 2026-11-11 at 4 24 54 PM" src="https://github.com/user-attachments/assets/73518897-50b7-4b7c-8cbc-9a45226e82f8" />

Validation results
<img width="1280" height="832" alt="WhatsApp Image 2026-11-11 at 4 24 55 PM" src="https://github.com/user-attachments/assets/1dcb80a5-d62f-4d84-92de-752415a018cf" />

Images showing predictions on the test set
<img width="1280" height="832" alt="test images" src="https://github.com/user-attachments/assets/5c95a827-1748-49af-aac1-5a65aa06068f" />


## 4. Recorded Training Environment

- Python: 3.13.16
- Ultralytics: 8.4.210
- PyTorch: 2.11.0+cu130
- Hardware: 2 x NVIDIA Tesla T4

Install PyTorch separately using a version compatible with your
target CUDA environment before installing the remaining packages.

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## 5. Dataset Structure

The training script expects the following dataset structure:

```text
pcb-defects/
    train/
        images/
        labels/
    valid/
        images/
        labels/
    test/
        images/
        labels/
```

The labels must use YOLO detection format:

```text
class_id x_center y_center width height
```

Coordinates must be normalized to the range 0 to 1.

## 6. Training Instructions

Run training from the submission root directory.

Example for a machine with two available GPUs:

```bash
python code/train.py \
    --dataset-root /path/to/pcb-defects \
    --preprocessed-root ./pcb_defects_clahe \
    --weights yolo26m.pt \
    --project ./pcb_runs \
    --device 0,1
```

For a machine with one GPU:

```bash
python code/train.py \
    --dataset-root /path/to/pcb-defects \
    --weights yolo26m.pt \
    --project ./pcb_runs \
    --device 0
```

Training uses CLAHE preprocessing, 896-pixel input resolution,
AdamW optimization, and the competition-specific mAP@50:90
checkpoint-selection metric.

## 7. Inference Instructions

Place the test images in a directory and run:

```bash
python code/inference.py \
    --weights weights/best.pt \
    --source /path/to/test/images \
    --output predictions.csv \
    --imgsz 896 \
    --conf 0.001 \
    --iou 0.70 \
    --max-det 300 \
    --device cpu
```

The inference script applies the same CLAHE preprocessing used
during training and runs detection on each image.

Use `--device 0` instead of `--device cpu` when GPU inference
is available and sufficient GPU memory is available.

## 8. Prediction CSV Format

The output CSV contains exactly these columns:

```text
image_name,class_name,x_min,y_min,x_max,y_max,confidence
```

Each row represents one predicted bounding box.

- `image_name`: Original image filename
- `class_name`: Predicted PCB defect class
- `x_min`, `y_min`: Top-left bounding-box coordinates
- `x_max`, `y_max`: Bottom-right bounding-box coordinates
- `confidence`: Prediction confidence between 0 and 1

Bounding-box coordinates are expressed in original image pixels.

## 9. Reproducibility

The training configuration, validation metrics, training history,
inference configuration, and generated prediction visualizations
are included where available in the submission package.

The final inference checkpoint is `weights/best.pt`.
