"""
@ Fridah Cheboi & Eric Hatungimana

This project implements a VTS technology to improve the quality of life 
of visually impaired people.
An image recognition model has been integrated with natural language processing model

Technical tools used include:
- Object Detection Algorithm - MobileNetV2 (replacing YOLOv5)
- Image recognition and classification - ResNet50
- Natural Language Processing - GPT-2
- Text to speech model - gTTS
- Web platform - streamlit
"""

# Import Necessary Libraries
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
from gtts import gTTS
import io
import cv2

import torch
from torchvision import transforms

# Load Pre-trained models
try:
    resnet_model = ResNet50(weights='imagenet')
except Exception as e:
    st.error(f"Failed to load model weights: {e}")
    st.stop()

# Load YOLOv7 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7-d6.pt').to(device)
yolo_model.eval()

# Define Helper Functions
def preprocess_image(image, model='resnet'):
    image = image.resize((224, 224))  # Resize for ResNet50
    img_array = img_to_array(image)
    if model == 'resnet':
        img_array = resnet_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_objects(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),  # Resize to match YOLO input size
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        detections = yolo_model(img_tensor)[0]
    
    detections = detections.cpu().numpy()
    
    class_ids = []
    confidences = []
    boxes = []
    
    for det in detections:
        for *box, conf, cls in det:
            if conf > 0.5:  # Confidence threshold
                box = np.array(box) * np.array([image.width, image.height, image.width, image.height])
                (x, y, w, h) = box.astype(int)
                x = int(x - (w / 2))
                y = int(y - (h / 2))
                boxes.append([x, y, w, h])
                confidences.append(float(conf))
                class_ids.append(int(cls))
    
    detections_df = pd.DataFrame({
        'xmin': [box[0] for box in boxes],
        'ymin': [box[1] for box in boxes],
        'xmax': [box[0] + box[2] for box in boxes],
        'ymax': [box[1] + box[3] for box in boxes],
        'confidence': confidences,
        'name': [str(class_id) for class_id in class_ids]
    })
    return detections_df

def generate_caption(classification, detections):
    objects = detections['name'].tolist()
    description = f"Generated description for an image with {', '.join(objects)} classified as {classification}."
    return description

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

# Implement Streamlit Web Interface
st.title("Vision to Speech Tool")
st.write("Upload an image or take a picture to get the description in text and audio.")

image_source = st.radio("Choose image source:", ("Upload", "Camera"))

if image_source == "Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif image_source == "Camera":
    picture = st.camera_input("Take a picture")
    if picture is not None:
        image = Image.open(picture)

if 'image' in locals():
    st.image(image, caption='Captured Image.', use_column_width=True)
    st.write("Processing...")
    
    # Preprocess and classify with ResNet50
    img_array = preprocess_image(image)
    preds = resnet_model.predict(img_array)
    classification = decode_predictions(preds, top=1)[0][0][1]
    
    # Detect objects with YOLOv7
    detections = detect_objects(image)
    
    # Generate caption
    caption = generate_caption(classification, detections)
    
    st.write(f"Classification: {classification}")
    st.write(f"Detected Objects: {', '.join(detections['name'].tolist())}")
    st.write(f"Generated Description: {caption}")
    
    # Convert text to speech
    audio_file = text_to_speech(caption)
    st.audio(audio_file, format='audio/mp3')
    
    # Display image with bounding boxes (simplified)
    img_np = np.array(image)
    img_pil = Image.fromarray(img_np)
    for _, obj in detections.iterrows():
        img_np = cv2.rectangle(img_np, 
                      (int(obj['xmin']), int(obj['ymin'])), 
                      (int(obj['xmax']), int(obj['ymax'])), 
                      (0, 255, 0), 2)
        img_np = cv2.putText(img_np, obj['name'], (int(obj['xmin']), int(obj['ymin'])-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    st.image(img_np, caption='Detected Objects', use_column_width=True)
