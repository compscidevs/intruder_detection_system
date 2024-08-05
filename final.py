
import serial
import serial.tools.list_ports
import time
import os
import cv2
import numpy as np
import tensorflow as tf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.utils import formataddr
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# Email configuration for Gmail
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'joshua.sj58@gmail.com'  # Replace with your Gmail address
EMAIL_PASSWORD = 'mheh qjkq qrrw odmd'  # Replace with your Gmail password or app password
OWNER_EMAIL = 'ssalijoshua2002@gmail.com'  # Replace with the owner's email address

# Load known face embeddings and labels
known_faces = {
    'person1': np.array([0.1, 0.2, 0.3]),  # Replace with actual embeddings
    # Add more known faces here
}

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # PIR sensor to GPIO4

# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error loading Haar Cascade XML file.")
else:
    print("Haar Cascade XML file loaded successfully.")

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_face(face_image):
    face_image = cv2.resize(face_image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    face_image = face_image.astype(np.uint8)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def get_face_embedding(face_image):
    preprocessed_face = preprocess_face(face_image)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_face)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    return embedding


def recognize_face(face_embedding, threshold=0.6):
    for name, known_embedding in known_faces.items():
        distance = np.linalg.norm(known_embedding - face_embedding)
        if distance < threshold:
            return name
    return "unknown"


def send_email(image_path):
    msg = MIMEMultipart()
    msg['From'] = formataddr(('Raspberry Pi', EMAIL_ADDRESS))
    msg['To'] = OWNER_EMAIL
    msg['Subject'] = 'Unknown Face Detected'

    with open(image_path, 'rb') as img_file:
        img = MIMEImage(img_file.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        msg.attach(img)

    text = MIMEText('An unknown face was detected. See the attached image.')
    msg.attach(text)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, OWNER_EMAIL, msg.as_string())

