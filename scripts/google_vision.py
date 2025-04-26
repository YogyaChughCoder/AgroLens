import tensorflow as tf
from google.cloud import vision
import numpy as np
from PIL import Image
import io
from test_model import aibro
from google.cloud import vision

def detect_with_vision_api(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    if response.error.message:
        print(f"Google Vision API error: {response.error.message}")
        return None
    labels = response.label_annotations
    if labels:
        return labels[0].description
    else:
        return "No disease detected by Vision API"

def predict_disease(image_path):
    try:
        predicted_disease=aibro(image_path)
        return predicted_disease
    except Exception as e:
        print(f"Model prediction failed: {e}")
        predicted_disease = detect_with_vision_api(image_path)
        print(f"Predicted by Google Vision API: {predicted_disease}")
        return predicted_disease


if __name__ == "__main__":
    image_path = "path_to_plant_image.jpg"
    disease = predict_disease('C:/Users/mudit/PycharmProjects/Agro_lens/data/test/Potato___Early_Blight/image (45).JPG')
    print(f"The disease detected is: {disease}")