
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
# Replace with your Gmail password or app password
EMAIL_PASSWORD = 'mheh qjkq qrrw odmd'
# Replace with the owner's email address
OWNER_EMAIL = 'ssalijoshua2002@gmail.com'

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
    face_image = cv2.resize(
        face_image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
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
        img.add_header('Content-Disposition', 'attachment',
                       filename=os.path.basename(image_path))
        msg.attach(img)

    text = MIMEText('An unknown face was detected. See the attached image.')
    msg.attach(text)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, OWNER_EMAIL, msg.as_string())


def list_serial_ports():
    """
    List all available serial ports.
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"Port: {port.device}")


def initialize_serial(port):
    """
    Initialize the serial connection.
    """
    return serial.Serial(port, baudrate=9600, timeout=1)


def send_at_command(ser, command, wait_for_response=True):
    """
    Send an AT command to the SIM800L module and optionally wait for the response.
    """
    ser.write((command + '\r\n').encode())
    if wait_for_response:
        time.sleep(1)  # wait for response
        response = ser.read_all().decode(errors='ignore')  # Ignore decode errors
        return response
    return None


def make_call(ser, phone_number):
    """
    Place a call to the specified phone number using the SIM800L module.
    """
    # Send the command to dial the number
    response = send_at_command(ser, 'ATD' + phone_number + ';')
    print(response)


def hang_up_call(ser):
    """
    Hang up the ongoing call.
    """
    response = send_at_command(ser, 'ATH')
    print(response)

motion_detected = False
last_motion_time = time.time()

# Specify the serial port
port = '/dev/ttyS0'  # Adjust this to your correct serial port
# Initialize serial connection
ser = initialize_serial(port)

try:
    while True:
        pir_state = GPIO.input(4)
        current_time = time.time()

        if pir_state == True:
            if not motion_detected:
                print('Motion Detected...')
                motion_detected = True
                last_motion_time = current_time
                picam2.start()
                print("Camera preview started.")

        if motion_detected:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            print(f"Detected faces: {len(faces)}")

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_image = frame[y:y+h, x:x+w]
                face_embedding = get_face_embedding(face_image)
                name = recognize_face(face_embedding)

                if name == "unknown":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    print("unknown user")
                    unknown_user_image_filename = "full_image.png"
                    cv2.imwrite(unknown_user_image_filename, frame)
                    send_email(unknown_user_image_filename)
                    
                    # Place a call after sending the email
                    phone_number = '0709735982'  # Replace with the phone number you want to call
                    print("Dialing...")
                    make_call(ser, phone_number)

                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Camera Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if current_time - last_motion_time > 60:
                print("No motion detected for 60 seconds. Turning off camera.")
                motion_detected = False
                picam2.stop()
                cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("Stopping camera preview...")

finally:
    GPIO.cleanup()
    if motion_detected:
        picam2.stop()
    cv2.destroyAllWindows()
    ser.close()
    print("Camera preview stopped.")

