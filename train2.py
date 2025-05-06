import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO



# Set paths
dataset_path = "dataset.yaml"  # Your existing YAML file
project_dir = "roof_detection"
run_name = "satellite_run1"

# Print current working directory
print(f"Current working directory: {os.getcwd()}")
print(f"Results will be saved to: {os.path.join(os.getcwd(), project_dir, run_name)}")

# Load a model
model = YOLO('yolo11s.pt')

# Train the model
results = model.train(
    data=dataset_path,
    epochs=100,
    imgsz=(1920, 1080),
    batch=8,
    patience=20,
    save=True,
    device="cpu",  # Change to 0 if you have GPU
    workers=4,
    project=project_dir,
    name=run_name,
    pretrained=True,
    optimizer='Adam',
    lr0=0.001,
    augment=True,
    verbose=True
)

# Print results location
results_dir = os.path.join(os.getcwd(), project_dir, run_name)
print(f"\n\nTraining completed!")
print(f"Results saved to: {results_dir}")
print(f"Best weights: {os.path.join(results_dir, 'weights/best.pt')}")
print(f"Last weights: {os.path.join(results_dir, 'weights/last.pt')}")

# Validate the model
model.val()