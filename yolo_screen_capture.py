from ultralytics import YOLO
import cv2
import numpy as np
import mss
import time
import ctypes
import os
import torch
import argparse
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict

# Parse command line arguments
parser = argparse.ArgumentParser(description="YOLO Screen Capture with Sliced Inference")
parser.add_argument("--model-path", type=str, help="Path to the model weights")
parser.add_argument("--width-ratio", type=float, default=0.5, help="Ratio of screen width to capture")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--sliced", action="store_true", help="Use sliced inference")
parser.add_argument("--slice-height", type=int, default=512, help="Height of slices")
parser.add_argument("--slice-width", type=int, default=512, help="Width of slices")
parser.add_argument("--overlap-height", type=float, default=0.2, help="Height overlap ratio")
parser.add_argument("--overlap-width", type=float, default=0.2, help="Width overlap ratio")
# Performance optimizations
parser.add_argument("--scale", type=float, default=1.0, help="Scale factor to resize image (0.5 = half size)")
parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip between processing")
args = parser.parse_args()

# Set device for inference
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load your trained model - update this path or use the specified path from command line
model_path = args.model_path
if not model_path:
    model_path = 'roof_detection/satellite_run/weights/best.pt'
    # Try to find the model if it doesn't exist at the default path
    if not os.path.exists(model_path):
        # Check if the directory exists
        base_dir = 'roof_detection'
        if os.path.exists(base_dir):
            # Find all run directories
            run_dirs = [d for d in os.listdir(base_dir) if d.startswith('satellite_run') and os.path.isdir(os.path.join(base_dir, d))]
            if run_dirs:
                # Find the latest run directory
                latest_run = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
                model_path = os.path.join(base_dir, latest_run, 'weights', 'best.pt')
                if not os.path.exists(model_path):
                    print(f"Could not find model at {model_path}")
                    model_path = 'yolov8n.pt'  # Fallback to pre-trained model
                    print(f"Using default YOLOv8n model: {model_path}")
            else:
                print("No run directories found.")
                model_path = 'yolov8n.pt'  # Fallback to pre-trained model
                print(f"Using default YOLOv8n model: {model_path}")

print(f"Loading model from: {model_path}")
model = YOLO(model_path).to(device)

# Get screen resolution
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Define left half for capture (adjust width as needed)
capture_width = int(screen_width * args.width_ratio)
monitor = {"top": 0, "left": 0, "width": capture_width, "height": screen_height}

# Initialize screen capture
sct = mss.mss()

# Set confidence threshold for detections
conf_threshold = args.conf

# Create SAHI detection model if using sliced inference
use_sliced = args.sliced
if use_sliced:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device
    )

# Create window and position it on the right side
window_name = "YOLO Roof Detection" + (" (Sliced)" if use_sliced else "")
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
frame_count = 0
skip_frames = args.skip_frames
last_annotated_frame = None

print("Screen capture started. Press 'q' to quit.")
print(f"Performance settings: Scale={args.scale}, Skip frames={skip_frames}")

try:
    while True:
        # Capture the left half of screen
        img = np.array(sct.grab(monitor))
        # Convert from BGRA to RGB
        img = img[:, :, :3]

        # Apply scaling if needed to improve performance
        if args.scale != 1.0:
            img_h, img_w = img.shape[:2]
            new_h, new_w = int(img_h * args.scale), int(img_w * args.scale)
            img = cv2.resize(img, (new_w, new_h))

        # Frame skipping logic for better performance
        frame_count += 1
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            # Show previous frame but skip processing
            if last_annotated_frame is not None:
                cv2.imshow(window_name, last_annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        # Process the frame
        annotated_frame = None
        if use_sliced:
            try:
                # Save the captured image temporarily
                temp_img_path = "temp_capture.jpg"
                cv2.imwrite(temp_img_path, img)

                # Run sliced inference
                result = get_sliced_prediction(
                    temp_img_path,
                    detection_model,
                    slice_height=args.slice_height,
                    slice_width=args.slice_width,
                    overlap_height_ratio=args.overlap_height,
                    overlap_width_ratio=args.overlap_width,
                )

                # Create a copy of the original image for visualization
                annotated_frame = img.copy()

                # Get object predictions
                object_predictions = result.object_prediction_list

                # Draw predictions on the image
                for prediction in object_predictions:
                    bbox = prediction.bbox.to_voc_bbox()
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    confidence = prediction.score.value
                    category_name = prediction.category.name if prediction.category.name else "object"

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    label = f"{category_name}: {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Clean up temp file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

            except Exception as e:
                print(f"Error in sliced inference: {e}")
                # Fall back to standard YOLO inference
                results = model(img, conf=conf_threshold)
                annotated_frame = results[0].plot()
        else:
            # Run standard inference on the captured screen
            results = model(img, conf=conf_threshold)
            annotated_frame = results[0].plot()

        # Store the annotated frame for frame skipping
        last_annotated_frame = annotated_frame.copy()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Add FPS to the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display additional info
        cv2.putText(annotated_frame, f"Scale: {args.scale:.1f}, Skip: {skip_frames}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow(window_name, annotated_frame)

        # Check if window has been closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Error: {e}")
finally:
    # Ensure cleanup
    if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
        try:
            os.remove(temp_img_path)
        except:
            pass
    cv2.destroyAllWindows()
    print("Screen capture stopped.")