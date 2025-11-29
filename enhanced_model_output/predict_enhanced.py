import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import json

def predict_disease(image_path):
    # Load model and labels
    model = tf.keras.models.load_model('enhanced_model.h5')
    
    with open('labels.json', 'r') as f:
        labels = json.load(f)
    
    # Preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    disease = labels[str(predicted_class)]
    
    print(f"Predicted Disease: {disease}")
    print(f"Confidence: {confidence:.2%}")
    
    return disease, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_enhanced.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_disease(image_path)
