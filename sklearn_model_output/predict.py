#!/usr/bin/env python3
"""
AgriSphere AI - Plant Disease Prediction Script (Scikit-learn version)
"""

import json
import numpy as np
from PIL import Image
import joblib
import sys

IMG_SIZE = 64

def extract_features(image_path):
    """
    Extract same features as used in training
    """
    try:
        # Open and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Extract features
        # 1. Color histogram features
        hist_r, _ = np.histogram(img_array[:,:,0], bins=32, range=(0, 256))
        hist_g, _ = np.histogram(img_array[:,:,1], bins=32, range=(0, 256))
        hist_b, _ = np.histogram(img_array[:,:,2], bins=32, range=(0, 256))
        
        # 2. Basic statistical features
        mean_rgb = np.mean(img_array, axis=(0, 1))
        std_rgb = np.std(img_array, axis=(0, 1))
        
        # 3. Flatten and combine features
        features = np.concatenate([
            hist_r, hist_g, hist_b,
            mean_rgb, std_rgb
        ])
        
        return features.reshape(1, -1)  # Reshape for prediction
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def predict_disease(image_path, model_path="model.pkl", labels_path="labels.json"):
    """
    Predict plant disease from image
    """
    # Load model and labels
    model = joblib.load(model_path)
    
    with open(labels_path, 'r') as f:
        class_names = json.load(f)
    
    # Extract features
    features = extract_features(image_path)
    if features is None:
        return None, None
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    predicted_class = class_names[prediction]
    confidence = probabilities[prediction]
    
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predicted_class, confidence = predict_disease(image_path)
    
    if predicted_class is None:
        print("Error: Could not process image")
        sys.exit(1)
    
    print(f"Predicted Disease: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
