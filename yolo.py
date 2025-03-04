#download_coco
from ultralytics.utils.downloads import download
from pathlib import Path

# Download labels
segments = False  # segment or box labels
dir = Path("Y:/CO")  # dataset root dir
url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'
urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
download(urls, dir=dir.parent)
# Download data
# urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
#       'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
#       'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
# download(urls, dir=dir / 'images', threads=3)

#delete_missing
import os

# Function to identify missing frames and delete the corresponding PNG files
def delete_missing_frame_images(label_folder, image_folder):
    # Get the list of .txt files in the label folder (only filenames, no full paths)
    label_files = {f.replace('.txt', '') for f in os.listdir(label_folder) if f.endswith(".txt")}
    
    # Get the list of .png files in the image folder (only filenames, no full paths)
    image_files = {f.replace('.png', '') for f in os.listdir(image_folder) if f.endswith(".png")}
    
    # Identify frames present in the image folder but missing in the label folder
    missing_frame_basenames = image_files - label_files

    # Delete corresponding PNG files for missing frames
    deleted_count = 0
    for frame in missing_frame_basenames:
        corresponding_png = frame + '.png'
        png_path = os.path.join(image_folder, corresponding_png)
        
        # Check if the corresponding PNG file exists
        if os.path.exists(png_path):
            os.remove(png_path)
            print(f"Deleted: {png_path}")
            deleted_count += 1
    
    print(f"Total deleted PNG files: {deleted_count}")

# Specify the paths to your label folder (with .txt files) and image folder (with .png files)
label_folder = "Y:/final/data/labels/train"  # Path to the folder with .txt files
image_folder = "Y:/final/data/images"  # Path to the folder with .png files

# Call the function
delete_missing_frame_images(label_folder, image_folder)

#delete_missing
# delete segmentation file
import os

# Paths to dataset
labels_dir ="Y:/data_1/coco_train/labels"  # Directory containing .txt label files
images_dir = "Y:/data_1/coco_train/images"  # Directory containing image files

# Clean annotations
for label_file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, "r") as f:
        lines = f.readlines()

    # Filter out lines with only bounding boxes (YOLO format: <class_id> <x_center> <y_center> <width> <height>)
    box_annotations = [line for line in lines if len(line.split()) == 5]

    # Overwrite the label file with box annotations only
    with open(label_path, "w") as f:
        f.writelines(box_annotations)

print("Segmentation annotations removed. Dataset ready for object detection.")

#change_id
#change old id to new id
import os

# Path to your custom annotations folder
annotations_path ="Y:/final/data/labels/train"

# Define the number of original YOLO classes
num_original_classes = 80  # e.g., COCO has 80 classes

# Create a mapping of old custom class IDs to new IDs
# Assuming your custom dataset had classes starting from 0
# If you had 5 custom classes, for example, their new IDs would be 80, 81, 82, etc.

old_to_new_class_map = {81:30,85:31,86:32}  # Mapping 0-74 to 33-107

# Iterate over all annotation files and remap class IDs
for filename in os.listdir(annotations_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(annotations_path, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Process each line (each line is a bounding box annotation)
        updated_lines = []
        for line in lines:
            elements = line.strip().split()
            class_id = int(elements[0])
            # Update class ID using the map
            if class_id in old_to_new_class_map:
                elements[0] = str(old_to_new_class_map[class_id])
            updated_lines.append(' '.join(elements))

        # Save the updated annotations
        with open(file_path, 'w') as file:
            file.write('\n'.join(updated_lines))

print("Class IDs remapped successfully!")

#change_id
a = ['person', 'bicycle', 'car', 'two wheeler', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 'sports ball', 'kite', 'potted plant', 'cell phone','auto','tractor','mini truck','all motor vehicle prohibited', 'axle load limit', 'bullock cart and hand cart prohibited', 'cattle ahead', 'chevron direction', 'compulsary ahead', 'compulsary ahead or turn left', 'compulsary ahead or turn right', 'compulsary keep left', 'compulsary keep right', 'compulsary sound horn', 'compulsary turn left ahead', 'compulsary turn right ahead', 'cross road', 'cycle crossing', 'cycle prohibited', 'dangerous dip', 'falling rocks', 'gap in median', 'give way', 'guarded level crossing', 'height limit', 'horn prohibited', 'hospital ahead', 'hump or rough road', 'left hand curve', 'left reverse bend', 'left turn prohibited', 'length limit', 'loose gravel', 'men at work', 'narrow bridge ahead', 'narrow road ahead', 'no entry', 'no parking', 'no stopping or standing', 'overtaking prohibited', 'pass either side', 'pedestrian crossing', 'pedestrian prohibited', 'petrol pump ahead', 'quay side or river bank', 'restriction ends', 'right hand curve', 'right reverse bend', 'right turn prohibited', 'road widens ahead', 'roundabout', 'school ahead', 'side road left', 'side road right', 'slippery road', 'speed limit 100', 'speed limit 120', 'speed limit 15', 'speed limit 20', 'speed limit 30', 'speed limit 40', 'speed limit 50', 'speed limit 60', 'speed limit 70', 'speed limit 80', 'staggered intersection', 'steep ascent', 'steep descent', 'stop', 'straight prohibited', 't intersection', 'traffic signal', 'truck prohibited', 'u turn', 'u turn prohibited', 'unguarded level crossing', 'width limit', 'y intersection' ]
count=0
for i in a:
    print(i,": ")
    print(count,"\n")
    count+=1

#xml_to_txt
import os
import xml.etree.ElementTree as ET

# Define class ID mapping
class_mapping = {
    'two_wheelers': 3,
    'vehicle_truck': 7,
    'auto': 30,
    'tractor': 31,
    'car': 2,
    'bicycle':1,
    'mini truck': 32,
    'tempo': 32,
    'bus': 5
}

def convert_xml_to_yolo(xml_folder, output_folder):
    """
    Convert Pascal VOC XML files to YOLO format .txt files.
    Args:
    - xml_folder: Path to the folder containing .xml annotation files.
    - output_folder: Path to save the converted .txt files in YOLO format.
    """
    os.makedirs(output_folder, exist_ok=True)

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract image size
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        # Prepare YOLO label file
        txt_file_name = os.path.splitext(xml_file)[0] + ".txt"
        txt_file_path = os.path.join(output_folder, txt_file_name)

        with open(txt_file_path, "w") as txt_file:
            # Process each object in the XML
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in class_mapping:
                    continue  # Skip classes not in the mapping

                class_id = class_mapping[class_name]
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                # Convert bbox to YOLO format
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # Write YOLO annotation
                txt_file.write(f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}\n")

        print(f"Converted: {xml_file} to {txt_file_name}")

# Paths
xml_folder = "Y:/tractor"  # Path to the folder containing .xml files
output_folder = "Y:/tractor-txt"  # Path to save the .txt files

# Run the conversion
convert_xml_to_yolo(xml_folder, output_folder)

#image_to_frames
import cv2
import os

def video_to_images(video_path, output_dir, fps=30):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open video file
    video_cap = cv2.VideoCapture(video_path)
    
    # Get original frame rate
    original_fps = video_cap.get(cv2.CAP_PROP_FPS)
    # Avoid division by zero by setting a minimum frame interval
    frame_interval = max(int(original_fps / fps), 1)
    
    print(f"Original FPS: {original_fps}")
    print(f"Frame interval: {frame_interval}")
    
    frame_count = 0
    save_count = 0
    
    # Read and save frames
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            break
        
        # Save frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = f"{output_dir}/frame_{save_count:06d}.png"
            cv2.imwrite(frame_filename, frame)
            save_count += 1
            print(f"Saved: {frame_filename}")
            
        frame_count += 1

    video_cap.release()
    print(f"Frames saved to {output_dir}")
#"C:\Users\STUDENT\Downloads\1732083497270583.mp4 && 1732083335490311  && IMG_0697"
# Usage
video_path ="C:/Users/STUDENT/Downloads/IMG_0697.mov"  # Replace with the path to your video file
output_dir = "Y:/final/data/images"  # Replace with your desired output directory
video_to_images(video_path, output_dir, fps=30)

#split_into_train
import os
import shutil
import random

# Paths
image_dir = "Y:/final/data/images"
label_dir = "Y:/final/data/labels/train"


train_img_dir = "Y:/final/3_tractor/train/images"
val_img_dir = "Y:/final/3_tractor/val/images"
test_img_dir ="Y:/final/3_tractor/test/images"

train_lbl_dir = "Y:/final/3_tractor/train/labels"
val_lbl_dir = "Y:/final/3_tractor/val/labels"
test_lbl_dir = "Y:/final/3_tractor/test/labels"

print('started')
# Ensure directories exist
for d in [train_img_dir, val_img_dir, test_img_dir, train_lbl_dir, val_lbl_dir, test_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Get list of image files
all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
random.shuffle(all_images)

# Split counts
train_count = int(0.7 * len(all_images))
val_count = int(0.2 * len(all_images))

# Move files
for idx, img_file in enumerate(all_images):
    label_file = img_file.replace('.png', '.txt')
    
    if idx < train_count:
        shutil.move(os.path.join(image_dir, img_file), os.path.join(train_img_dir, img_file))
        shutil.move(os.path.join(label_dir, label_file), os.path.join(train_lbl_dir, label_file))
    elif idx < train_count + val_count:
        shutil.move(os.path.join(image_dir, img_file), os.path.join(val_img_dir, img_file))
        shutil.move(os.path.join(label_dir, label_file), os.path.join(val_lbl_dir, label_file))
    else:
        shutil.move(os.path.join(image_dir, img_file), os.path.join(test_img_dir, img_file))
        shutil.move(os.path.join(label_dir, label_file), os.path.join(test_lbl_dir, label_file))
    print(f"Moved {img_file} to {train_img_dir if idx < train_count else val_img_dir if idx < train_count + val_count else test_img_dir}")

print("Dataset split into train, val, and test folders.")

#yolo_train_1.0
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. No GPU detected.")

#yolo_train_1.0
import torch
torch.cuda.empty_cache()

#yolo_train_1.0
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11x.pt")  # Update the path to your desired model checkpoint

# Train the model on your custom dataset using the GPU (device=0)
results = model.train(data="Y:/final/data.yaml",
                      epochs=200,
                      optimizer='AdamW',
                      save_period=15,
                      patience=20,
                      imgsz=640,
                      workers=24,
                      device=0,
                     freeze = 12,
                    batch=0.60,                           # Batch size
                    pretrained=True,                    # Use pretrained weights
                    lr0=0.001,                         # Learning rate for AdamW
                    weight_decay=0.01,                 # Regularization
                    augment=True,                      # Enable augmentations
                    mosaic=True,                       # Enable mosaic augmentation
                    flipud=0.5,                        # Vertical flip probability
                    fliplr=0.5,                        # Horizontal flip probability
                    hsv_h=0.015,                      # Adjust hue
                    hsv_s=0.7,                        # Adjust saturation
                    hsv_v=0.4
                     )  # Use GPU 0 (NVIDIA GeForce RTX 3090)

#yolo_train_1.0
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11x.pt")  # Update the path to your desired model checkpoint

# Train the model on your custom dataset using the GPU (device=0)
results = model.train(data="Y:/final/data.yaml",  # path to your custom dataset YAML file
                      epochs=50,
                      optimizer='AdamW',
                      save_period=15,
                      patience=20,
                      imgsz=640,
                      workers=24,
                      device=0,
                      batch=0.60,
                     freeze = 0)  # Use GPU 0 (NVIDIA GeForce RTX 3090)

from ultralytics import YOLO

# Load a model
#model=YOLO("yolo11m.pt")
model = YOLO("runs/detect/train38/weights/best.pt")  # pretrained YOLO11n model
#print(model)
# Run batched inference on a list of images
results = model.val()
3

results.box.maps
#results.show()

from ultralytics import YOLO

# Load a model
#model=YOLO("yolo11m.pt")
model = YOLO("runs/detect/train40/weights/last.pt")  # pretrained YOLO11n model
#print(model)
# Run batched inference on a list of images
results = model.val()
results.box.maps
#results.show()
