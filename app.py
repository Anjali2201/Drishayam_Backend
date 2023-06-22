from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the TFLite model
model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the name and age from the request
    name = request.form.get('name')
    age = request.form.get('age')
    image_path = request.files["image"]
    image_path.save('./Assets/uploaded_image.jpg')

    image_new = "./Assets/uploaded_image.jpg"
    image = tf.keras.preprocessing.image.load_img(image_new, target_size=(55, 94))
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
    prediction = float(output_data[0])
    # convert into percentages with two decimal points
    prediction_new = round(prediction * 100, 2)


    # Create the response payload
    status=""
    if prediction > 0.5:
        status="Normal Vision"
    else :
        status="Cataract"
    response = {
        'prediction': {
            'name': name,
            'age': age,
            'status': status,
            'probability': prediction_new,
        }
    }
    # Return the response as JSON
    print (response)
    return jsonify(response)



# @app.route('/predict2', methods=['POST'])
# def predict2():
#     # Get the name and age from the request
#     name = request.form.get('name')
#     age = request.form.get('age')

#     # Get the image file from the request
#     image = request.files['image']
#     image.save('uploaded_image.jpg')

#     # Print the received data
#     print(f"Name: {name}")
#     print(f"Age: {age}")


#     return 'Success'

if __name__ == '__main__':
    app.run(
        debug=True,
    )
