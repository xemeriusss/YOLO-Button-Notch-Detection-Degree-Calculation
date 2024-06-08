from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import math
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QPushButton, QFileDialog, QListWidget, QListWidgetItem, 
                             QHBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
# import os
# HOME = os.getcwd()
# print(HOME)

# Define the source and destination directories
source_dir = 'images/'
dest_dir = 'resized_images/'

# Create the destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Define the new size
new_size = (640, 640)  # YOLOv5 default size, you can adjust as needed

# Resize and save each image
for img_name in os.listdir(source_dir):
    img_path = os.path.join(source_dir, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        resized_img = cv2.resize(img, new_size)
        cv2.imwrite(os.path.join(dest_dir, img_name), resized_img)

# Load the model
model = YOLO('best.pt')

# Make predictions
predictions = model(source=dest_dir, save=True)

# Create labels directory if it doesn't exist
labels_dir = Path("labels")
labels_dir.mkdir(parents=True, exist_ok=True)

# Loop through each image in predictions
for prediction in predictions:
    image_path = Path(prediction.path)
    label_path = labels_dir / f"{image_path.stem}.txt"

    try:
        # Write the first bounding box to the label file
        if len(prediction.boxes) > 0:
            box = prediction.boxes.xywhn[0]
            cls = int(prediction.boxes.cls[0].item())
            x, y, w, h = box.tolist()
            with open(label_path, 'w') as file:
                file.write(f"{cls} {x} {y} {w} {h}\n")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

print("Label files created successfully.")




def load_annotations(label_path):
    annotations = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Assuming label format: class x_center y_center width height
            # Convert from YOLO format to absolute coordinates (x, y)
            x, y = float(parts[1]), float(parts[2])
            annotations.append((x, y))
    return annotations


def calculate_degree(image, annotations):
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2

    degrees = []
    for (x, y) in annotations:
        # Convert relative to absolute coordinates
        notch_x, notch_y = x * width, y * height

        # Calculate the angle relative to the center
        angle = math.atan2(notch_y - center_y, notch_x - center_x)
        degree = math.degrees(angle)

        if degree < 0:
            degree += 360

        # Adjust degree for image orientation
        degree = (degree + 90) % 360
        formatted_value = f"{degree:.3f}"
        degrees.append(formatted_value)

    return degrees



class DegreeViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Degree Viewer')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.layout = QVBoxLayout()
        
        self.header_label = QLabel("Degree Viewer", self)
        self.header_label.setFont(QFont('Arial', 24))
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("color: #333333; margin: 10px;")
        self.layout.addWidget(self.header_label)
        
        self.load_button = QPushButton('Load Images and Labels')
        self.load_button.setFont(QFont('Arial', 14))
        self.load_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.load_button.clicked.connect(self.load_files)
        self.layout.addWidget(self.load_button)
        
        self.list_widget = QListWidget()
        self.list_widget.setFont(QFont('Arial', 12))
        self.list_widget.setStyleSheet("background-color: #ffffff; border: 1px solid #dddddd; border-radius: 5px;")
        self.list_widget.itemClicked.connect(self.display_image_and_degrees)
        self.layout.addWidget(self.list_widget)
        
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #ffffff; border: 1px solid #dddddd; border-radius: 5px; margin-top: 10px;")
        self.layout.addWidget(self.image_label)
        
        self.degrees_label = QLabel(self)
        self.degrees_label.setFont(QFont('Arial', 14))
        self.degrees_label.setAlignment(Qt.AlignCenter)
        self.degrees_label.setStyleSheet("color: #333333; margin-top: 10px;")
        self.layout.addWidget(self.degrees_label)
        
        self.setLayout(self.layout)
    
    def load_files(self):
        self.image_dir = QFileDialog.getExistingDirectory(self, 'Select Image Directory')
        self.label_dir = QFileDialog.getExistingDirectory(self, 'Select Label Directory')
        
        if self.image_dir and self.label_dir:
            self.process_files(self.image_dir, self.label_dir)
    
    def process_files(self, image_dir, label_dir):
        self.list_widget.clear()
        
        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))

                image = cv2.imread(image_path)
                annotations = load_annotations(label_path)
                degrees = calculate_degree(image, annotations)
                
                item_text = f'File: {filename}, Degrees of Notches: {", ".join(map(str, degrees))}'
                list_item = QListWidgetItem(item_text)
                list_item.setData(Qt.UserRole, (image_path, degrees))
                self.list_widget.addItem(list_item)
    
    def display_image_and_degrees(self, item):
        image_path, degrees = item.data(Qt.UserRole)
        
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio))
        
        degrees_text = f"Degrees of Notches: {', '.join(degrees)}"
        self.degrees_label.setText(degrees_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DegreeViewer()
    viewer.show()
    sys.exit(app.exec_())