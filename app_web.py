from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from yolo_pred import YOLO_Pred

app = Flask(__name__)

# Initialize YOLO
yolo = YOLO_Pred("./Model/weights/best.onnx", 'data.yaml')
def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, closest_warning = yolo.predictions(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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

                return render_template('index_web.html', output_image=output_path, input_image=original_image_path, closest_warning=closest_warning)

    if not closest_warning:
        closest_warning = "Looks safe, keep moving"  # Display "Looks safe" when no pedestrians are close

    return render_template('index_web.html', output_image=None, input_image=None, closest_warning=closest_warning)

if __name__ == '__main__':
    app.run(debug=True)
