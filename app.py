from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

def convert_to_sketch(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image
    inverted_gray_image = 255 - gray_image
    # Blur the inverted grayscale image
    blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    # Invert the blurred image
    inverted_blurred_image = 255 - blurred_image
    # Create the final sketch image by blending the grayscale image with the inverted blurred image
    sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
    return sketch_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return 'Invalid file', 400

    sketch_image = convert_to_sketch(image)

    # Convert sketch image to base64 for sending to frontend
    retval, buffer = cv2.imencode('.jpg', sketch_image)
    sketch_base64 = base64.b64encode(buffer).decode('utf-8')

    # Optionally, save the sketch image as a file
    cv2.imwrite('sketch.jpg', sketch_image)

    return jsonify({'sketch': sketch_base64})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
