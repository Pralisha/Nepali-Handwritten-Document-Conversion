import streamlit as st
from ocr import perform_ocr
from yolo_utils import detect_words_in_image, crop_word_images, draw_bounding_boxes, save_cropped_images, setup_image_directory
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor
import torch
import io
import os
import config

# Directory to save cropped images
CROPPED_IMAGE_DIR = "cropped_images"


def get_images_from_directory(directory):
    """Get all images from the local directory and group them by line number."""
    images = {}

    # List all files in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg"):
            # Extract the line number from filename (e.g., '1_1.jpg' -> '1')
            line_number = filename.split('_')[0]

            # Open the image
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)

            # Group images by line number
            if line_number not in images:
                images[line_number] = []
            # Store both image and filename
            images[line_number].append((img, filename))

    return images


st.markdown(
    """
    <style>
    .reportview-container {
        background: white;  /* Change background color to white */
    }
    .stText, .stMarkdown {
        color: black;  /* Change text color to black for contrast */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Nepali Handwritten Document Conversion")

uploaded_image = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the YOLO model using the path from config
    results = detect_words_in_image(image, config.YOLO_MODEL_PATH)
    word_images = crop_word_images(image, results)

    # Draw bounding boxes on the image
    image_with_boxes = image.copy()  # Make a copy of the original image
    image_with_boxes = draw_bounding_boxes(image_with_boxes, results)

    # Display the image with bounding boxes
    st.image(image_with_boxes, caption="Image with Bounding Boxes",
             use_column_width=True)

    # Set up and save cropped images
    setup_image_directory()
    save_cropped_images(word_images)

    # Load the OCR model using the path from config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ocr_model = VisionEncoderDecoderModel.from_pretrained(
        config.OCR_MODEL_PATH)
    ocr_model.to(device)

    # Prepare feature extractor and tokenizer using paths from config
    feature_extractor = ViTFeatureExtractor.from_pretrained(config.VIT_ENCODER)
    tokenizer = RobertaTokenizer.from_pretrained(config.ROBERTA_DECODER)
    processor = TrOCRProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Show loading spinner while performing OCR
    with st.spinner("Processing... Please wait."):
        final_output, word_mappings = perform_ocr(
            word_images, processor, ocr_model, device, tokenizer)

    # Show the compiled text in the Streamlit app
    st.subheader("Compiled Text:")
    st.text(final_output)  # Display the entire compiled text

    # Create a download button for the compiled text
    compiled_text_buffer = io.BytesIO()
    compiled_text_buffer.write(final_output.encode('utf-8'))
    compiled_text_buffer.seek(0)  # Move to the start of the BytesIO object

    st.download_button(
        label="Download Compiled Text",
        data=compiled_text_buffer,
        file_name="compiled_text.txt",
        mime="text/plain"
    )

    if st.button("Show More Details"):
        # Organize word_mappings by line numbers
        line_images = {}
        for img, predicted_word, img_file in word_mappings:
            # Extract the line number from the filename, assumed to be like '1_1.jpg'
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            line_number = img_name.split('_')[0]  # Keep line_number as string

            if line_number not in line_images:
                # Create a new list for this line
                line_images[line_number] = []

            # Append tuple of (image, predicted word)
            line_images[line_number].append((img, predicted_word))

        st.subheader("Cropped Images and Predicted Words Displayed Line-Wise:")
        # Sort line numbers to maintain order
        for line_number in sorted(line_images.keys()):
            # Create columns equal to the number of images in this line
            cols = st.columns(len(line_images[line_number]))

            for col, (img, predicted_word) in zip(cols, line_images[line_number]):
                with col:
                    # Resize the cropped image for display
                    resized_image = img.resize(
                        (150, int(150 * img.height / img.width)))  # Maintain aspect ratio
                    st.image(resized_image, caption=f"{predicted_word}")
