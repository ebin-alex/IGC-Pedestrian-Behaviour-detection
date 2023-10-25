import math
import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader
from math import sqrt
import Jetson.GPIO as GPIO
import time

class YOLO_Pred:
    def __init__(self, onnx_model, data_yaml):
        with open('data.yaml', mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        # Rough estimation of lane width on Indian roads in pixels
        lane_width_pixels = 350  # Adjust this value based on your estimation
        
        # Rough estimation of lane width on Indian roads in centimeters
        lane_width_cm = 350  # Adjust this value based on your estimation
        
        # Calculate the scale factor for distance estimation
        self.scale_factor = lane_width_pixels / lane_width_cm
        self.yolo = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Use CUDA backend for GPU acceleration
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # Use FP16 for even faster inference
        self.confidence_threshold = 0.2  # You can adjust this threshold

    def angle_to_bottom_center(self, x, y, w, h, image_w, image_h):
        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h

        # Calculate the angle between the bottom center of the image and the center of the bounding box
        angle = math.atan2(center_x - image_w // 2, image_h - center_y)
        
        return angle
    
    def calculate_distance(self, x, y, w, h, image_w, image_h):
        # Calculate the angle between the bottom center of the image and the center of the bounding box
        angle = math.atan2((x + w / 2) - (image_w / 2), self.camera_focal_length)

        # Calculate the distance in meters using the angle and camera focal length
        distance_meters = self.actual_lane_width_meters / (2 * math.tan(angle / 2))

        return distance_meters

    def find_closest_person(self, boxes, distances, angles):
        if distances:
            closest_distance = min(distances)
            closest_index = distances.index(closest_distance)
        else:
            closest_index = -1
        
        return closest_index

    def generate_colors(self):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return colors

    def predictions(self, image):
        GPIO.setmode(GPIO.BOARD)
        
        self.red_led = 12
        self.yellow_led = 7
        self.green_led = 11
        self.motor = 31
        
        GPIO.setup(self.red_led, GPIO.OUT)
        GPIO.setup(self.yellow_led, GPIO.OUT)
        GPIO.setup(self.green_led, GPIO.OUT)
        GPIO.setup(self.motor,GPIO.OUT)

        GPIO.output(self.motor, GPIO.LOW)

        row, col, d = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        input_wh_yolo = 640
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (input_wh_yolo, input_wh_yolo), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        text_color = (0, 255, 0) 
        # Filter detections based on confidence score (0.2) and probability score (0.25)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / input_wh_yolo
        y_factor = image_h / input_wh_yolo

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.2:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # Construct bounding box from four values
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # Non-maximum suppression
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.15, 0.25)

        distances = []
        angles_text = []

        for ind in index:
            x, y, w, h = boxes_np[ind]

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h

            # Calculate the angle to the bottom center and add it to angles_text
            angle = self.angle_to_bottom_center(x, y, w, h, image_w, image_h)
           
                
            if angle < 0:
                angle_text = "Left"
            elif angle >= 0:
                angle_text = "Right" 
            elif angle< 150 and angle< 90:
                angle_text = "Middle"   
            
            # Calculate the distance in meters using the scale factor
            distance_pixels = sqrt((center_x - image_w // 2) ** 2 + (center_y - image_h) ** 2)
            distance_meters = distance_pixels / (self.scale_factor * 100)  # Convert from centimeters to meters
            distances.append(distance_meters)

            angles_text.append(angle_text)

        
        closest_warning = ""  # Initialize closest_warning here
        colors = self.generate_colors()

        red_detected = False
        yellow_detected = False
        green_detected = False

        for i, ind in enumerate(index):
            x, y, w, h = boxes_np[ind]
            bb_conf = int(confidences_np[ind] * 100)
            classes_id = classes[ind]
            class_name = self.labels[classes_id]

    # Calculate the angle to the bottom center and add it to angles_text
            angle = self.angle_to_bottom_center(x, y, w, h, image_w, image_h)

            if angle < 0:
                angle_text = "Left"
            elif angle >= 0:
                angle_text = "Right"
            else:
                angle_text = "Middle"

            # Calculate the distance in meters using the scale factor
            distance_pixels = sqrt((center_x - image_w // 2) ** 2 + (center_y - image_h) ** 2)
            distance_meters = distance_pixels / (self.scale_factor * 100)  # Convert from centimeters to meters
            distances.append(distance_meters)
            # Draw the bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), colors[classes_id], 2)
            cv2.rectangle(image, (x, y - 30), (x + w, y), colors[classes_id], -1)

            # Add text for class, distance, and angle at the top
            # Add text for class, distance, and angle at the top
           

            # Add text for distance in bold letters on top of the bounding box
            distance_text = f'  {distances[i]:.2f} m'
            text_x = x + w // 2 - len(distance_text) * 6  # Adjust the X-coordinate based on text length
            text_y = y + h + 20  # Adjust the Y-coordinate as needed
            cv2.putText(image, distance_text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)


            # Add text for angle at the bottom
            text_x = x
            text_y = y + h + 45  # Adjust this value as needed to control the vertical position of the text
            text_color = (0, 0, 255)
            angle_text = angles_text[i]  # Use angles_text directly
            cv2.putText(image, angle_text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 1)
 
            if i < len(distances):  # Make sure i is within the valid range
                if distances[i] <= 7:
                    # Display a warning message for the closest person within 7 meters
                    closest_warning = f"The person on the {angle_text} is too close!"
                    text_color = (0, 0, 255)  # Set text color to red

                    # Change the color of the bounding box to red
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 0), -1)
                    red_detected = True
                elif distances[i] > 7 and distances[i] <= 10:
                    # Change the color of the bounding box to yellow
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 165, 0), 2)
                    cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 165, 0), -1)
                    closest_warning = f"Careful!"
                    yellow_detected = True
                else:
                    # No warning message for safe distances
                    closest_warning = "Looks safe, keep moving"
                    text_color = (0, 255, 0)  # Set text color to green

                    # Change the color of the bounding box to green
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(image, (x, y - 30), (x + w, y), (0, 255, 0), -1)
                    green_detected = True
        # Draw the "Looks safe, keep moving" or other warning text
        text_x = 10  # Adjust the X-coordinate as needed
        text_y = 30  # Adjust the Y-coordinate as needed
        cv2.putText(image, closest_warning, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, text_color, 2)

        # Convert the image to BGR format for saving
        img_pred = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if red_detected:
            GPIO.output(self.red_led, GPIO.HIGH)
            GPIO.output(self.yellow_led, GPIO.LOW)
            GPIO.output(self.green_led, GPIO.LOW)
            GPIO.output(self.motor, GPIO.LOW)
        elif yellow_detected:
            GPIO.output(self.red_led, GPIO.LOW)
            GPIO.output(self.yellow_led, GPIO.HIGH)
            GPIO.output(self.green_led, GPIO.LOW)
            GPIO.output(self.motor, GPIO.LOW)
        else:
            GPIO.output(self.red_led, GPIO.LOW)
            GPIO.output(self.yellow_led, GPIO.LOW)
            GPIO.output(self.green_led, GPIO.HIGH)
            GPIO.output(self.motor, GPIO.HIGH)
            time.sleep(2)  # Delay for 5 seconds
            GPIO.output(self.motor, GPIO.LOW)

        return img_pred, closest_warning
    

