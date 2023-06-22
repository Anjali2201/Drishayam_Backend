from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the TFLite model
model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']

    # Load and preprocess the input image
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(54, 94))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image /= 255.0  # Normalize the image

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])


    # Example: Assuming the output is a single value representing the prediction
    prediction = output_data[0]

    # Create the response payload
    response = {'prediction': prediction}

    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run()
