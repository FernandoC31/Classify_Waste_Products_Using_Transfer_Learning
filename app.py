from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import base64

app = Flask(__name__)

model = load_model('Classifier_Waste.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    img = Image.open(file.stream)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    img = img.resize((150, 150))  
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  
    
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        Predictions = "Residuo Reciclable"
    else:
        Predictions = "Residuo Orgánico"

    return render_template('index.html', prediction=Predictions, uploaded_image=img_str)

if __name__ == '__main__':
    app.run(debug=True)
