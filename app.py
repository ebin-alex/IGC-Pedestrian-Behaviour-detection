from flask import Flask, render_template, request
import cv2
import numpy as np
from yolo_predictions import YOLO_Pred

app = Flask(__name__)

# Initialize YOLO
yolo = YOLO_Pred("./Model/weights/best.onnx", 'data.yaml')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image file upload
        if 'image' in request.files:
            image = request.files['image']
            if image:

                # Read the uploaded image
                img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                
                # Create a copy of the original image for display
                original_img = img.copy()
                
                img_pred = yolo.predictions(img)
                output_path = 'static/output_image.png'
                cv2.imwrite(output_path, img_pred)
                
                # Original image path
                original_image_path = 'static/input_image.png'
                cv2.imwrite(original_image_path, original_img)
                
                
                return render_template('index.html', output_image=output_path, input_image=original_image_path)

    return render_template('index.html', output_image=None, input_image=None)


if __name__ == '__main__':
    app.run(debug=True)
