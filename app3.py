from flask import Flask, render_template,request, Response
import cv2
import numpy as np
from yolo_pred import YOLO_Pred

app = Flask(__name__)

# Initialize YOLO
yolo = YOLO_Pred("./Model/weights/best.onnx", 'data.yaml')

def detect_pedestrians():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Set a lower resolution for faster processing
    cap.set(3, 640)  # Set the width
    cap.set(4, 480)  # Set the height

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (320, 240))
        img_pred, closest_warning = yolo.predictions(frame)  # Get closest_warning

        ret, jpeg = cv2.imencode('.jpg', img_pred)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
@app.route('/', methods=['GET', 'POST'])
def index():
    closest_warning = ""  # Initialize closest_warning

    if request.method == 'POST':
        # Handle image file upload
        if 'image' in request.files:
            image = request.files['image']
            if image:
                # Read the uploaded image
                img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)

                # Create a copy of the original image for display
                original_img = img.copy()

                img_pred, closest_warning = yolo.predictions(img)  # Get closest_warning

                output_path = 'static/output_image.png'
                cv2.imwrite(output_path, img_pred)

                # Original image path
                original_image_path = 'static/input_image.png'
                cv2.imwrite(original_image_path, original_img)

                return render_template('index2.html', output_image=output_path, input_image=original_image_path, closest_warning=closest_warning)

    if not closest_warning:
        closest_warning = "Looks safe, keep moving"  # Display "Looks safe" when no pedestrians are close

    return render_template('index_web.html', output_image=None, input_image=None, closest_warning=closest_warning)
@app.route('/video')
def video():
    return Response(detect_pedestrians(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
