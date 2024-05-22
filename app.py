from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Mapping between predicted class indices and nutrient deficiencies
def get_deficiency(class_index):
    deficiency_mapping = {
        0: 'Healthy',
        1: 'Nitrogen',
        2: 'Phosphorus',
        3: 'Potassium',
    }
    return deficiency_mapping.get(class_index, 'Unknown')

# Load the trained model
def load_model():
    model = tf.keras.models.load_model("DN169_2.keras")
    return model

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict(image, model):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    prediction_probabilities = predictions[0]
    return class_index, prediction_probabilities

@app.route("/predict", methods=["POST"])
def predict_deficiency():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        model = load_model()
        class_index, prediction_probabilities = predict(image, model)
        deficiency_type = get_deficiency(class_index)
        
        prediction_table_data = {}
        for i, prob in enumerate(prediction_probabilities):
            class_name = get_deficiency(i)
            prediction_table_data[class_name] = f"{prob:.2f}"

        response = {
            "deficiency_type": deficiency_type,
            "prediction_probabilities": prediction_table_data
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
