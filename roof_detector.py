"""
Roof Detection - Integrated Project
----------------------------------
This project combines training and inference for roof detection using YOLO.
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
    def __init__(self):
        self.project_dir = "roof_detection"
        self.dataset_path = "dataset.yaml"
        self.model_path = None
        self.model = None

    def train(self, epochs=100, run_name="satellite_run", pretrained_model='yolo11n.pt'):
        """Train the YOLO model for roof detection"""
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        print(f"Current working directory: {os.getcwd()}")

        # Load the base model
        model = YOLO(pretrained_model)

        # Train the model
        results = model.train(
            data=self.dataset_path,
            epochs=epochs,
            imgsz=1080,
            batch=4,
            patience=20,
            save=True,
            device="cpu",  # Change to 'cuda' for GPU training
            workers=0,
            project=self.project_dir,
            name=run_name,
            pretrained=True,

            # Heavy augmentation for small dataset
            augment=True,
            degrees=20.0,
            translate=0.3,
            scale=0.7,
            fliplr=0.5,
            flipud=0.3,
            mosaic=1.0,
            mixup=0.2,

            # Use higher cls loss weight for better class balance
            box=7.5,     # Box loss gain
            cls=3.0,     # Cls loss gain

            # Prevent overfitting
            dropout=0.2,

            verbose=True
        )

        # Get the path where model was saved
        results_dir = os.path.join(os.getcwd(), self.project_dir, run_name)
        self.model_path = os.path.join(results_dir, 'weights', 'best.pt')

        print(f"\nTraining completed!")
        print(f"Results saved to: {results_dir}")
        print(f"Best weights: {self.model_path}")

        # Load and validate the trained model
        self._validate_model()

        return self.model_path

    def _validate_model(self):
        """Validate the trained model and run a test prediction"""
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"Warning: Model file not found at {self.model_path}")
            return False

        self.model = YOLO(self.model_path)
        metrics = self.model.val()
        print(f"Validation metrics: mAP50-95={metrics.box.map:.4f}, mAP50={metrics.box.map50:.4f}")

        # Try to find test images
        try:
            test_dir = os.path.join(os.getcwd(), "collect-ai", "datasets", "my-bina", "images", "val")
            if os.path.exists(test_dir):
                test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                               if f.endswith(('.jpg', '.jpeg', '.png'))]
                if test_images:
                    test_img = test_images[0]
                    results = self.model(test_img, conf=0.25)
                    pred_path = os.path.join(os.path.dirname(self.model_path), 'sample_prediction.jpg')
                    results[0].save(pred_path)
                    print(f"Sample prediction saved to: {pred_path}")
        except Exception as e:
            print(f"Could not process sample image: {e}")

        return True

    def load_model(self, model_path=None):
        """Load an existing trained model"""
        if model_path:
            self.model_path = model_path

        if not self.model_path:
            print("Error: No model path specified")
            return False

        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False

        self.model = YOLO(self.model_path)
        print(f"Model loaded successfully from {self.model_path}")
        return True

    def screen_capture(self, conf_threshold=0.25, capture_width_ratio=0.5):
        """Run the model on screen capture for real-time inference"""
        if not self.model:
            print("Error: No model loaded. Please load a model first.")
            return

        # Get screen resolution
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        # Define capture area (left portion of screen)
        capture_width = int(screen_width * capture_width_ratio)
        monitor = {"top": 0, "left": 0, "width": capture_width, "height": screen_height}

        # Initialize screen capture
        sct = mss.mss()

        # Create window and position it on the right side
        window_name = "Roof Detection - YOLO"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Calculate window position and size for right half
        window_x = capture_width + 10  # Add a small gap
        window_width = screen_width - capture_width - 20  # Leave some margin
        window_height = int(window_width * screen_height / capture_width)  # Maintain aspect ratio

        # Position window on right side
        cv2.moveWindow(window_name, window_x, 0)
        cv2.resizeWindow(window_name, window_width, window_height)

        # For FPS calculation
        prev_time = time.time()

        print("Starting screen capture. Press 'q' to quit.")

        try:
            while True:
                # Capture the screen
                img = np.array(sct.grab(monitor))
                # Convert from BGRA to RGB
                img = img[:, :, :3]

                # Run inference on the captured screen
                results = self.model(img, conf=conf_threshold)

                # Visualize results
                annotated_frame = results[0].plot()

                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                # Add FPS to the frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the annotated frame
                cv2.imshow(window_name, annotated_frame)

                # Check if window has been closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Screen capture interrupted")
        finally:
            # Ensure cleanup
            cv2.destroyAllWindows()
            print("Screen capture stopped")

    def sliced_inference(self, image_path, conf_threshold=0.25, slice_height=512, slice_width=512,
                         overlap_height_ratio=0.2, overlap_width_ratio=0.2):
        """Run sliced inference on an image using SAHI"""
        if not self.model:
            print("Error: No model loaded. Please load a model first.")
            return None

        # Create SAHI detection model from the YOLO model
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=conf_threshold,
            device="cpu" if not torch.cuda.is_available() else "cuda:0"
        )

        # Perform sliced prediction
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        return result

    def sliced_screen_capture(self, conf_threshold=0.25, capture_width_ratio=0.5,
                             slice_height=512, slice_width=512,
                             overlap_height_ratio=0.2, overlap_width_ratio=0.2):
        """Run the model on screen capture using sliced inference for real-time detection"""
        if not self.model:
            print("Error: No model loaded. Please load a model first.")
            return

        # Create SAHI detection model from the YOLO model
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=conf_threshold,
            device="cpu" if not torch.cuda.is_available() else "cuda:0"
        )

        # Get screen resolution
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        # Define capture area
        capture_width = int(screen_width * capture_width_ratio)
        monitor = {"top": 0, "left": 0, "width": capture_width, "height": screen_height}

        # Initialize screen capture
        sct = mss.mss()

        # Create window and position it on the right side
        window_name = "Roof Detection - YOLO (Sliced)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Calculate window position and size for right half
        window_x = capture_width + 10
        window_width = screen_width - capture_width - 20
        window_height = int(window_width * screen_height / capture_width)

        # Position window on right side
        cv2.moveWindow(window_name, window_x, 0)
        cv2.resizeWindow(window_name, window_width, window_height)

        # For FPS calculation
        prev_time = time.time()

        print("Starting sliced screen capture. Press 'q' to quit.")

        try:
            while True:
                # Capture the screen
                img = np.array(sct.grab(monitor))
                # Convert from BGRA to RGB
                img = img[:, :, :3]

                # Save the captured image temporarily
                temp_img_path = "temp_capture.jpg"
                cv2.imwrite(temp_img_path, img)

                # Run sliced inference
                result = get_sliced_prediction(
                    temp_img_path,
                    detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                )

                # Create a copy of the original image for visualization
                visualization = img.copy()
                
                # Get object predictions
                object_predictions = result.object_prediction_list
                
                # Draw predictions on the image
                for prediction in object_predictions:
                    bbox = prediction.bbox.to_voc_bbox()
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    confidence = prediction.score.value
                    category_name = prediction.category.name if prediction.category.name else "object"
                    
                    # Draw bounding box
                    cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{category_name}: {confidence:.2f}"
                    cv2.putText(visualization, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                # Add FPS to the frame
                cv2.putText(visualization, f"FPS: {fps:.1f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add slice info
                cv2.putText(visualization, f"Slice: {slice_width}x{slice_height}, Overlap: {overlap_width_ratio:.1f}", 
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display the annotated frame
                cv2.imshow(window_name, visualization)

                # Check if window has been closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Screen capture interrupted")
        except Exception as e:
            print(f"Error in sliced screen capture: {e}")
        finally:
            # Ensure cleanup
            if os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except:
                    pass
            cv2.destroyAllWindows()
            print("Screen capture stopped")


def main():
    parser = argparse.ArgumentParser(description="Roof Detection with YOLO")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--run-name", type=str, default="satellite_run", help="Name for this training run")
    train_parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model to start with")

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference on screen capture")
    inference_parser.add_argument("--model-path", type=str, help="Path to the trained model weights")
    inference_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    inference_parser.add_argument("--width-ratio", type=float, default=0.5,
                                help="Ratio of screen width to capture (0-1)")
    inference_parser.add_argument("--sliced", action="store_true", help="Use sliced inference")
    inference_parser.add_argument("--slice-height", type=int, default=512, help="Height of slices")
    inference_parser.add_argument("--slice-width", type=int, default=512, help="Width of slices")
    inference_parser.add_argument("--overlap-height", type=float, default=0.2, help="Height overlap ratio")
    inference_parser.add_argument("--overlap-width", type=float, default=0.2, help="Width overlap ratio")

    args = parser.parse_args()

    # Initialize the detector
    detector = RoofDetector()

    if args.command == "train":
        detector.train(epochs=args.epochs, run_name=args.run_name, pretrained_model=args.model)

    elif args.command == "inference":
        if args.model_path:
            model_loaded = detector.load_model(args.model_path)
        else:
            # Try to find the most recent model
            project_dir = Path(detector.project_dir)
            if project_dir.exists():
                runs = [d for d in project_dir.iterdir() if d.is_dir() and d.name.startswith("satellite_run")]
                if runs:
                    latest_run = max(runs, key=lambda x: x.stat().st_mtime)
                    model_path = latest_run / "weights" / "best.pt"
                    if model_path.exists():
                        print(f"Found latest model at {model_path}")
                        model_loaded = detector.load_model(str(model_path))
                    else:
                        print("Could not find a model weights file. Please specify with --model-path")
                        return
                else:
                    print("No training runs found. Please train a model first or specify a model path")
                    return
            else:
                print(f"Project directory {project_dir} not found. Please train a model first")
                return

        if detector.model:
            if args.sliced:
                detector.sliced_screen_capture(
                    conf_threshold=args.conf,
                    capture_width_ratio=args.width_ratio,
                    slice_height=args.slice_height,
                    slice_width=args.slice_width,
                    overlap_height_ratio=args.overlap_height,
                    overlap_width_ratio=args.overlap_width
                )
            else:
                detector.screen_capture(conf_threshold=args.conf, capture_width_ratio=args.width_ratio)

    else:
        # If no command is specified, show help
        parser.print_help()


if __name__ == "__main__":
    main()