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