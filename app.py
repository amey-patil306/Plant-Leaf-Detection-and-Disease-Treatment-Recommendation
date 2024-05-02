import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
import cv2

app = Flask(__name__)

# Load the pre-trained model
cnn = tf.keras.models.load_model('model.keras')

# Define class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def preprocess_image(img):
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image to match model input shape
    img = cv2.resize(img, (128, 128))
    # Scale pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the POST request
        img_file = request.files['file']
        
        # Read image as OpenCV object
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), -1)
        
        # Preprocess the image for prediction
        input_img = preprocess_image(img)
        
        # Perform prediction
        predictions = cnn.predict(input_img)
        
        # Get the predicted class index
        predicted_index = np.argmax(predictions[0])
        
        # Get the predicted class name
        predicted_class = class_names[predicted_index]
        
        return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
