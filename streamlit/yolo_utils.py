# yolo_utils.py
from ultralytics import YOLO
from PIL import ImageDraw
import os
import shutil

# Directory to save cropped images
CROPPED_IMAGE_DIR = "cropped_images"

def setup_image_directory():
    """Create directory if not exists, and empty if it already exists."""
    if os.path.exists(CROPPED_IMAGE_DIR):
        # Empty the directory
        shutil.rmtree(CROPPED_IMAGE_DIR)
    os.makedirs(CROPPED_IMAGE_DIR)

def save_cropped_images(word_images):
    """Save the cropped word images line-wise in the local directory."""
    for word_image, img_file in word_images:
        img_path = os.path.join(CROPPED_IMAGE_DIR, f"{img_file}.jpg")
        word_image.save(img_path)
        
def detect_words_in_image(image, model_path):
    model = YOLO(model_path)
    results = model(image)  # Run YOLOv8 inference on the image
    return results

def crop_word_images(image, results, vertical_threshold=10):
    word_images = []
    
    if isinstance(results, list):
        results = results[0]  # Get the first results object
    
    boxes = results.boxes.xyxy  # Get the bounding boxes
    boxes_list = [box.tolist() for box in boxes]
    sorted_boxes = sorted(boxes_list, key=lambda b: (b[1] + b[3]) / 2)

    lines = []
    current_line = []

    for box in sorted_boxes:
        avg_ymin = (box[1] + box[3]) / 2
        if not current_line:
            current_line.append(box)
        else:
            prev_box_avg_ymin = (current_line[-1][1] + current_line[-1][3]) / 2
            if abs(avg_ymin - prev_box_avg_ymin) <= vertical_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
    
    if current_line:
        lines.append(current_line)
    
    for line_index, line in enumerate(lines, start=1):
        line_sorted = sorted(line, key=lambda b: b[0])
        for word_index, box in enumerate(line_sorted, start=1):
            xmin, ymin, xmax, ymax = map(int, box)
            cropped_word = image.crop((xmin, ymin, xmax, ymax))
            word_images.append((cropped_word, f'{line_index}_{word_index}'))
    
    return word_images



def draw_bounding_boxes(image, results):
    """Draw bounding boxes on the image using YOLO results."""
    if isinstance(results, list):
        results = results[0]

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Draw each bounding box on the image
    for box in results.boxes.xyxy:
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)  # Red box, 2px thick

    return image