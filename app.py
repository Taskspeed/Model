from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def get_deficiency(class_index):
    deficiency_mapping = {
        0: 'Healthy',
        1: 'Nitrogen',
        2: 'Phosphorus',
        3: 'Potassium',
    }
    return deficiency_mapping.get(class_index, 'Unknown')

def load_model():
    model = tf.keras.models.load_model("DN169_2.keras")
    return model

def preprocess_image(image):
    image = image.convert('RGB')
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    return img_array

def predict(image, model):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    prediction_probabilities = predictions[0]
    return class_index, prediction_probabilities

@app.route("/result", methods=["POST"])
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
        
        # Get the probability for the detected deficiency type
        accuracy = prediction_probabilities[class_index] * 100  # Convert to percentage
        
        # Prepare the response data
        response_data = {
            'deficiency_type': deficiency_type,
            'accuracy': f"{accuracy:.2f}%"  # Format as percentage with two decimal points
        }

        # Convert image to base64 for display
        image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_data = base64.b64encode(buffered.getvalue()).decode()

        # return jsonify({
        #     'deficiency_type': deficiency_type,
        #     'accuracy': response_data['accuracy'],
        #     'img_data': img_data
        # })
        return jsonify({
        'deficiency_type': deficiency_type,
        'accuracy': f"{accuracy:.2f}%",  # Ensure accuracy is returned in response
        'img_data': img_data
                })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host=host, debug=debug)
    # app.run(host='192.168.100.105', debug=True)
