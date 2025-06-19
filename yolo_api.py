from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import io

app = Flask(__name__)
model = YOLO("D:/projects mobile apps/My Apps/leg_deformities/best2.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    results = model.predict(img)
    preds = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0]) * 100
            preds.append({'class': name, 'confidence': round(conf, 2)})

        # Draw boxes on image
        annotated_img = result.plot()

    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', annotated_img)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'predictions': preds,
        'image': base64_img
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)