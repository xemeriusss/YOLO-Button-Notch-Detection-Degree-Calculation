from ultralytics import YOLO

# Load pre-trained model
model = YOLO('best.pt')

# Run inference on a single image
results = model(source="home_oven.jpg", save=True)