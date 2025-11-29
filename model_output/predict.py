#!/usr/bin/env python3
"""
AgriSphere AI - Plant Disease Prediction Script
Load trained model and predict disease from plant images
"""

import json
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

class PlantDiseasePredictor:
    def __init__(self, model_path="model_output/model.h5", labels_path="model_output/labels.json"):
        self.model = tf.keras.models.load_model(model_path)
        
        with open(labels_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        print(f"Model loaded with {len(self.class_mapping)} classes")
        print(f"Classes: {list(self.class_mapping.values())}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path):
        """Predict disease from image"""
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get class name
        predicted_class = self.class_mapping[str(predicted_class_idx)]
        
        return predicted_class, confidence
    
    def predict_with_details(self, image_path):
        """Predict with detailed probabilities for all classes"""
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        results = []
        for idx, prob in enumerate(predictions):
            class_name = self.class_mapping[str(idx)]
            results.append((class_name, prob))
        
        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize predictor
    predictor = PlantDiseasePredictor()
    
    # Make prediction
    predicted_class, confidence = predictor.predict(image_path)
    
    print(f"\nPrediction Results:")
    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # Show top 3 predictions
    print(f"\nTop 3 Predictions:")
    detailed_results = predictor.predict_with_details(image_path)
    for i, (class_name, prob) in enumerate(detailed_results[:3]):
        print(f"   {i+1}. {class_name}: {prob:.2%}")
