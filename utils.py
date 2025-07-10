# utils.py
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict_burn(model, image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    pred_class = np.argmax(preds)
    confidence = np.max(preds)
    return pred_class, confidence
