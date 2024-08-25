from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

app = Flask(__name__)

# Load your trained model
from model import create_model  # Assuming create_model() creates your model architecture
model = create_model()
model.load_weights('my_model_weights.h5')

# Define your class labels
class_labels = ['Normal', 'Oral Cancer']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        
        # Preprocess the image
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]

        # Display prediction result
        return render_template('result.html', prediction=predicted_class_label)

if __name__ == '__main__':
    app.run(debug=True)
