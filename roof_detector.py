"""
Roof Detection - Integrated Project
----------------------------------
This project combines training and inference for roof detection using YOLO and SAHI sliced inference.
"""

import os
import argparse
import numpy as np
import cv2
import time
import mss
import ctypes
import torch
from pathlib import Path
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class RoofDetector:
    def __init__(self,
                 project_dir: str = "roof_detection",
                 dataset_path: str = "dataset.yaml"):
        self.project_dir = project_dir
        self.dataset_path = dataset_path
        self.model_path = None
        self.model = None

    def train(self,
              epochs: int = 150,
              run_name: str = "satellite_run",
              pretrained_model: str = "yolo11n.pt",
              optimizer: str = "adams",
              lr0: float = 0.02,
              lrf: float = 0.01,
              patience: int = 30,
              batch: int = 4,
              imgsz: int = 1080) -> str:
        """Train the YOLO model for roof detection"""
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        print(f"Current working directory: {os.getcwd()}")

        model = YOLO(pretrained_model)
        results = model.train(
            data=self.dataset_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=patience,
            save=True,
            workers=0,
            project=self.project_dir,
            name=run_name,
            pretrained=True,

            # Optimizer settings
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,


            # Data augmentation (in-memory)
            augment=True,
            degrees=20.0,
            translate=0,
            scale=0.7,
            shear=0.0,
            perspective=0.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,
            cutmix=0.0,
            copy_paste=0.0,
            auto_augment='randaugment',
            erasing=0.4,

            # Loss component weights
            cls=1.5,
            dfl=1.5,
            pose=12.0,
            kobj=2,
            box= 7.5,
        )

        weights_dir = Path(self.project_dir) / run_name / 'weights'
        self.model_path = str(weights_dir / 'best.pt')

        print(f"\nTraining completed! Best weights saved to: {self.model_path}")
        self._validate_model()
        return self.model_path

    def _validate_model(self) -> bool:
        """Validate the trained model and save a sample prediction"""
        if not self.model_path or not Path(self.model_path).exists():
            print(f"Warning: Model file not found at {self.model_path}")
            return False
        self.model = YOLO(self.model_path)
        metrics = self.model.val()
        print(f"Validation metrics: mAP50-95={metrics.box.map:.4f}, mAP50={metrics.box.map50:.4f}")

        test_dir = Path("collect-ai/datasets/my-bina/images/val")
        if test_dir.exists():
            imgs = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if imgs:
                res = self.model(str(imgs[0]), conf=0.25)
                sample_path = Path(self.model_path).parent / 'sample_prediction.jpg'
                res[0].save(str(sample_path))
                print(f"Sample prediction saved to: {sample_path}")
        return True

    def load_model(self, model_path: str) -> bool:
        """Load a trained YOLO model from disk"""
        if not Path(model_path).exists():
            print(f"Error: Model file not found at {model_path}")
            return False
        self.model_path = model_path
        self.model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        return True

    def screen_capture(self,
                       conf_threshold: float = 0.25,
                       width_ratio: float = 0.5,
                       scale: float = 1.0,
                       skip_frames: int = 0):
        """Run real-time inference on screen capture"""
        if not self.model:
            print("Error: No model loaded. Use load_model() first.")
            return

        user32 = ctypes.windll.user32
        sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        cap_w = int(sw * width_ratio)
        monitor = {"top": 0, "left": 0, "width": cap_w, "height": sh}
        sct = mss.mss()

        win = "Roof Detection"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.moveWindow(win, cap_w + 10, 0)

        prev = time.time()
        last = None

        print("Starting screen capture. Press 'q' to quit.")
        while True:
            img = np.array(sct.grab(monitor))[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            if scale != 1.0:
                h, w = img.shape[:2]
                img = cv2.resize(img, (int(w*scale), int(h*scale)))

            # Frame skipping
            if skip_frames and last is not None:
                if last:
                    cv2.imshow(win, last)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

            res = self.model(img, conf=conf_threshold)
            frame = res[0].plot()

            now = time.time()
            fps = 1 / (now - prev)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(win, frame)
            last = frame.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Screen capture stopped.")

    def sliced_screen_capture(self,
                               conf_threshold: float = 0.25,
                               width_ratio: float = 0.5,
                               slice_h: int = 512,
                               slice_w: int = 512,
                               overlap_h: float = 0.2,
                               overlap_w: float = 0.2):
        """Run real-time sliced inference on screen capture"""
        if not self.model:
            print("Error: No model loaded. Use load_model() first.")
            return
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        det_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=conf_threshold,
            device=device
        )

        user32 = ctypes.windll.user32
        sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        cap_w = int(sw * width_ratio)
        monitor = {"top": 0, "left": 0, "width": cap_w, "height": sh}
        sct = mss.mss()

        win = "Roof Detection (Sliced)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.moveWindow(win, cap_w + 10, 0)

        prev = time.time()
        temp = "temp_cap.jpg"

        print("Starting sliced screen capture. Press 'q' to quit.")
        while True:
            img = np.array(sct.grab(monitor))[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            cv2.imwrite(temp, img)
            result = get_sliced_prediction(
                temp, det_model,
                slice_height=slice_h, slice_width=slice_w,
                overlap_height_ratio=overlap_h,
                overlap_width_ratio=overlap_w
            )
            frame = img.copy()
            for pred in result.object_prediction_list:
                x1, y1, x2, y2 = map(int, pred.bbox.to_voc_bbox())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{pred.category.name}:{pred.score.value:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            now = time.time()
            fps = 1 / (now - prev)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(win, frame)
            if os.path.exists(temp):
                os.remove(temp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Sliced capture stopped.")


def main():
    parser = argparse.ArgumentParser(description="Roof Detection with YOLO")
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("train", help="Train the model")
    p1.add_argument("--epochs", type=int, default=150)
    p1.add_argument("--run-name", type=str, default="satellite_run")
    p1.add_argument("--model", type=str, default="yolov8n.pt")
    p1.add_argument("--optimizer", type=str, default="auto")
    p1.add_argument("--lr0", type=float, default=0.01)
    p1.add_argument("--lrf", type=float, default=0.01)
    p1.add_argument("--patience", type=int, default=20)
    p1.add_argument("--batch", type=int, default=4)
    p1.add_argument("--imgsz", type=int, default=1080)

    p2 = sub.add_parser("inference", help="Run real-time inference")
    p2.add_argument("--model-path", type=str)
    p2.add_argument("--conf", type=float, default=0.25)
    p2.add_argument("--width-ratio", type=float, default=0.5)
    p2.add_argument("--scale", type=float, default=1.0)
    p2.add_argument("--skip-frames", type=int, default=0)

    p3 = sub.add_parser("sliced", help="Run sliced screen capture")
    p3.add_argument("--model-path", type=str)
    p3.add_argument("--conf", type=float, default=0.25)
    p3.add_argument("--width-ratio", type=float, default=0.5)
    p3.add_argument("--slice-h", type=int, default=512)
    p3.add_argument("--slice-w", type=int, default=512)
    p3.add_argument("--overlap-h", type=float, default=0.2)
    p3.add_argument("--overlap-w", type=float, default=0.2)

    args = parser.parse_args()
    rd = RoofDetector()

    if args.cmd == "train":
        rd.train(
            epochs=args.epochs,
            run_name=args.run_name,
            pretrained_model=args.model,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz
        )
    elif args.cmd == "inference":
        if rd.load_model(args.model_path or rd.model_path):
            rd.screen_capture(
                conf_threshold=args.conf,
                width_ratio=args.width_ratio,
                scale=args.scale,
                skip_frames=args.skip_frames
            )
    elif args.cmd == "sliced":
        if rd.load_model(args.model_path or rd.model_path):
            rd.sliced_screen_capture(
                conf_threshold=args.conf,
                width_ratio=args.width_ratio,
                slice_h=args.slice_h,
                slice_w=args.slice_w,
                overlap_h=args.overlap_h,
                overlap_w=args.overlap_w
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
