from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Load the trained model weights
try:
    model.load_weights('C:/Users/theja/OneDrive/Desktop/FIRSTAPP/backend/my_model_weights.weights.h5')
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        return None, f"Error preprocessing image: {e}"

def predict(image_path):
    try:
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is None:
            return None, "Error preprocessing image"
        
        prediction = model.predict(preprocessed_image)
        result = "Oral cancer detected." if prediction[0][0] > 0.5 else "No oral cancer detected."
        return result, None
    except Exception as e:
        return None, f"Error predicting: {e}"

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        print("Recieved1")
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        result, error = predict(file_path)
        os.remove(file_path)  # Clean up the saved file
        
        if error:
            return jsonify({"error": error}), 500
        else:
            return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(host='0.0.0.0', port=5000)
