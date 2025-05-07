# Roof Detection for MYZ303 Homework

![Roof Detection Demo](https://raw.githubusercontent.com/username/Roof-Detection-MYZ303-Homework/main/demo/detection_demo.jpg)

## 📋 Project Overview

This project, developed for the MYZ303 course at Istanbul Technical University, implements roof detection and classification using YOLO (You Only Look Once) models. It supports both satellite imagery training and real-time inference via screen capture, with optional sliced inference for large images.

### 🎯 Goals

* Train YOLO models to detect and classify regular building roofs and mosque roofs in satellite imagery.
* Compare model performance between hand-annotated datasets and synthetically augmented datasets.
* Analyze the impact of data augmentation techniques.
* Provide real-time detection through screen-capture-based inference with optional slice-based processing.

## 📦 Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Roof-Detection-MYZ303-Homework.git
   cd Roof-Detection-MYZ303-Homework
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare dataset**

   * Organize images and labels:

    ```text
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
    ├── train/
    └── val/
    ```
   * Create `dataset.yaml`:

    ```bash
    cp dataset.yaml.sample dataset.yaml
    ```
   * Edit `dataset.yaml` paths and class names:

    ```yaml
    path: ./dataset
    train: images/train
    val: images/val
    names:
      0: bina
      1: cami
    ```

## 🧰 Scripts & Usage

### 1. Training with `roof_detector.py`

This script trains a YOLO model with heavy augmentation and saves the best weights.

```bash
python roof_detector.py train   --epochs <int>    # Number of epochs to train (default: 150)
  --run-name <string>    # Directory name for this run (default: satellite_run)
  --model <path>    # Pretrained YOLO weights (default: yolo11n.pt)
  --optimizer <string>   # Optimizer to use (default: auto)
  --lr0 <float>     # Initial learning rate (default: 0.01)
  --lrf <float>     # Final learning rate factor (default: 0.01)
  --patience <int>  # Early stopping patience (default: 20)
  --batch <int>     # Batch size (default: 4)
  --imgsz <int>     # Image size (default: 1080)
```

**Example**

```bash
python roof_detector.py train --epochs 150 --run-name my_model --model yolo11n.pt
```

| Parameter    | Type  | Default    | Description    |
| ---- | ---- | ---- | ---- |
| `--epochs`   | `int` | `150`    | Total number of training epochs.    |
| `--run-name` | `str` | `satellite_run` | Name of the training run; outputs stored under `roof_detection/<run-name>/`. |
| `--model`    | `str` | `yolo11n.pt`    | Path to the base YOLO model for fine-tuning.    |
| `--optimizer`| `str` | `auto`    | Optimizer to use for training.    |
| `--lr0`      | `float` | `0.01`    | Initial learning rate.    |
| `--lrf`      | `float` | `0.01`    | Final learning rate factor.    |
| `--patience` | `int` | `20`    | Early stopping patience.    |
| `--batch`    | `int` | `4`    | Batch size for training.    |
| `--imgsz`    | `int` | `1080`    | Image size for training.    |

After training, the best weights are saved under `roof_detection/<run-name>/weights/best.pt` and automatically validated (mAP metrics printed). A sample prediction image is also generated if test images are found.

### 2. Inference & Screen Capture with `roof_detector.py`

Run real-time detection on the left half of your screen:

```bash
python roof_detector.py inference   --model-path <path>    # Path to trained weights (default: finds latest)     --conf <float>    # Confidence threshold (0.0–1.0, default: 0.25)      --width-ratio <float>   # Portion of screen to capture (0.0–1.0, default: 0.5)   --scale <float>   # Scale factor for performance (default: 1.0)   --skip-frames <int>   # Frames to skip between detections (default: 0)
```

**Example**

```bash
python roof_detector.py inference --model-path roof_detection/my_model/weights/best.pt --conf 0.3 --width-ratio 0.6
```

| Parameter    | Type    | Default | Description    |
| ---- | ---- | ---- | ---- |
| `--model-path`    | `str`   | (auto)  | Path to `.pt` weights. If omitted, loads latest run. |
| `--conf`    | `float` | `0.25`  | Minimum confidence for rendering detection boxes.    |
| `--width-ratio`    | `float` | `0.5`   | Fraction of screen width to capture from left.    |
| `--scale`    | `float` | `1.0`   | Image scaling factor to speed up inference (e.g., 0.5 = half size). |
| `--skip-frames`    | `int`   | `0`    | Number of frames to skip between inference passes.    |

### 3. Sliced Inference with `roof_detector.py`

For large images or detailed detection, use sliced inference:

```bash
python roof_detector.py sliced   --model-path <path>    # Path to trained weights (default: finds latest)     --conf <float>    # Confidence threshold (0.0–1.0, default: 0.25)      --width-ratio <float>   # Portion of screen to capture (0.0–1.0, default: 0.5)   --slice-h <int>  # Slice height in pixels (default: 512)      --slice-w <int>   # Slice width in pixels (default: 512)      --overlap-h <float> # Vertical overlap ratio (0.0–1.0, default: 0.2)
  --overlap-w <float>  # Horizontal overlap ratio (0.0–1.0, default: 0.2)
```

**Example**

```bash
python roof_detector.py sliced --model-path roof_detection/my_model/weights/best.pt --conf 0.3 --width-ratio 0.6 --slice-h 600 --slice-w 600
```

| Parameter    | Type    | Default | Description    |
| ---- | ---- | ---- | ---- |
| `--model-path`    | `str`   | (auto)  | Path to `.pt` weights. If omitted, loads latest run. |
| `--conf`    | `float` | `0.25`  | Minimum confidence for rendering detection boxes.    |
| `--width-ratio`    | `float` | `0.5`   | Fraction of screen width to capture from left.    |
| `--slice-h`   | `int`   | `512`   | Height (px) of image slices in sliced inference.    |
| `--slice-w`    | `int`   | `512`   | Width (px) of image slices in sliced inference.    |
| `--overlap-h` | `float` | `0.2`   | Vertical overlap ratio between slices.    |
| `--overlap-w`  | `float` | `0.2`   | Horizontal overlap ratio between slices.    |

### 4. Quick Start with `run_detection.bat`

For Windows users, a batch file is provided for easy execution:

```
run_detection.bat [model_path] [width_ratio] [conf_threshold] [sliced] [slice_height] [slice_width] [overlap_height] [overlap_width] [scale] [skip_frames]
```

**Example**

```
run_detection.bat best.pt 0.5 0.3 true 512 512 0.2 0.2 0.8 1
```

All parameters are optional with sensible defaults.

## 🧪 Model Comparison & Analysis

You can train two configurations and compare mAP metrics:

```bash
# Hand-annotated only
python roof_detector.py train --epochs 150 --run-name hand_annotated

# Hand-annotated + synthetic
python roof_detector.py train --epochs 150 --run-name synthetic_augmented
```

View validation metrics printed at training end, or run:

```bash
# Visualize class predictions
python class_visualization.py --model roof_detection/hand_annotated/weights/best.pt --output results/hand
```

## 🙏 Acknowledgements

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [SAHI (Sliced Aided Hyper Inference)](https://github.com/obss/sahi)
* [collect-ai](https://github.com/itumekanik/collect-ai)
* MYZ303 course at Istanbul Technical University

## 📄 License

Licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
