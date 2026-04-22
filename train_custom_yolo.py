import os
import urllib.request
import shutil
import logging
from ultralytics import YOLO

import numpy as np
import cv2

"""
YOLOv8 Custom Training Workflow for Grain Detection.
This script automates synthetic dataset generation, configuration (data.yaml),
and model training for Rice and Pepper classifications.
"""

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 1. Setup Dataset Directory Structure
dataset_dir: str = os.path.abspath("sample_rice_pepper_dataset")
dirs: list[str] = ['images/train', 'images/val', 'labels/train', 'labels/val']
logging.info(f"Initializing dataset structure at: {dataset_dir}")
for d in dirs:
    os.makedirs(os.path.join(dataset_dir, d), exist_ok=True)

# 3. Generate Synthetic Images & Create Bounding Boxes
logging.info("Generating synthetic training assets (Rice & Pepper benchmarks)...")
# We use synthetic images (colored squares) for demo purposes to avoid internet download blocks.
colors = {0: (200, 200, 200), 1: (50, 50, 50)} # Rice (light), Pepper (dark)

for class_id in range(2):
    for i in range(3):
        img_name = f"class_{class_id}_img_{i}.jpg"
        lbl_name = f"class_{class_id}_img_{i}.txt"
        
        img_train = os.path.join(dataset_dir, "images", "train", img_name)
        lbl_train = os.path.join(dataset_dir, "labels", "train", lbl_name)
        img_val = os.path.join(dataset_dir, "images", "val", img_name)
        lbl_val = os.path.join(dataset_dir, "labels", "val", lbl_name)
        
        # Generate a 320x320 image with the specific color in the center
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        # Background color
        img[:] = (100, 150, 100)
        # Grain color
        cv2.rectangle(img, (60, 60), (260, 260), colors[class_id], -1) 
        cv2.imwrite(img_train, img)
        
        # Create a bounding box (class, x_center, y_center, width, height)
        # The rectangle above goes from 60 to 260, out of 320.
        # Center = 160/320 = 0.5. Width/Height = 200/320 = 0.625
        with open(lbl_train, "w") as f:
            f.write(f"{class_id} 0.5 0.5 0.625 0.625\n")
        
        # Duplicate to validation set
        shutil.copy(img_train, img_val)
        shutil.copy(lbl_train, lbl_val)
        logging.info(f"[Done] Generated {img_name}")

# 4. Create data.yaml mapping file
yaml_content = f"""path: {dataset_dir}
train: images/train
val: images/val
nc: 2
names: ['Rice', 'Pepper']
"""
yaml_path = os.path.join(dataset_dir, "data.yaml")
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

logging.info(f"\nDataset fully prepared at: {dataset_dir}")
logging.info("===================================================")
logging.info("🚀 STARTING YOLOv8 CUSTOM TRAINING PIPELINE")
logging.info("===================================================")

# 5. Execute Training
logging.info("Initializing neural network training on local device.")
model = YOLO("yolov8n.pt") # Start with base nano model
results = model.train(
    data=yaml_path, 
    epochs=5,           # Small epochs for demo speed
    imgsz=320,          # Smaller resolution for speed
    batch=2,            
    device='cpu',       # Universal compatibility
    plots=False         # Skip plot generation to save time
)

# 6. Extract Best Model
run_dir = model.trainer.save_dir
best_model_path = os.path.join(run_dir, "weights", "best.pt")
final_model_dest = os.path.abspath("custom_rice_pepper_model.pt")

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, final_model_dest)
    logging.info("===================================================")
    logging.info(f"✅ TRAINING COMPLETE! Model saved as:\n{final_model_dest}")
    logging.info("===================================================")
else:
    logging.warning("⚠️ Training finished but could not locate best.pt!")
