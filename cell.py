import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model_path = r'C:\Users\vinoth\Desktop\Main project\main-project\models\LeukemiaModel.keras'
model = tf.keras.models.load_model(model_path)

# Path to the image
img_path = r'C:\Users\vinoth\Desktop\Main project\main-project\beg.jpg'

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

print(f'Predicted Class: {predicted_class}')
