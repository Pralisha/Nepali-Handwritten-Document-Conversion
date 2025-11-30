# Nepali OCR Implementation

This repository provides an implementation for OCR (Optical Character Recognition) of Nepali handwritten documents using a combination of YOLO for word detection and a fine-tuned TrOCR model for text recognition.


## Installation

1. **Clone the repository**:

```bash
   https://github.com/Pralisha/Nepali-Handwritten-Document-Conversion.git
```
2. **Install required libraries:**

```bash
  pip install -r requirements.txt
```
## Configuration

1. **Download YOLO model and TrOCR checkpoints:**
  - Download the YOLO model (best.pt) and the pre-trained TrOCR model checkpoint directory from (https://drive.google.com/drive/folders/1Gk1zjmod0YzFLSoyaSnuJa6SINAJBhCN?usp=sharing) and place them in the appropriate directories.

2. **Update paths:**
  - Open the config.py file and update the folder paths for the YOLO model and TrOCR checkpoints.


## Running the Application

**Launch the Streamlit app:**

The OCR system is accessible through a Streamlit web app. To run the app, use the following command:

```bash
streamlit run app_1.py   
```
## Usage

**Once the app is running, follow these steps:**

1. Upload a scanned image of a Nepali handwritten document.
2. The app will process the image, detect words, and perform OCR to extract the text.
3. The processed text will be displayed, and cropped images will be shown row-wise, maintaining their original order.
4. The compiled text can be downloaded in text format.
