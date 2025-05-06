from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Train the model with specific parameters for satellite imagery
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,  # resize images to 640x640
    batch=8,    # batch size
    patience=20,  # early stopping patience
    save=True,  # save checkpoints
    device="cpu",   # use CPU
    workers=4,  # number of worker threads
    project='roof_detection',  # project name
    name='satellite_run1',  # experiment name
    pretrained=True,  # use pretrained weights
    optimizer='Adam',  # optimizer
    lr0=0.001,  # initial learning rate
    augment=True  # use data augmentation
)
