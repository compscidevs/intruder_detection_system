import cv2
import mediapipe as mp
import numpy as np
import face_recognition
from scipy.spatial import distance
from picamera2 import Picamera2
import os
import time

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

# Load face embeddings and corresponding labels
face_embeddings = np.load("face_embeddings.npy", allow_pickle=True).item()
face_labels = np.load("face_labels.npy", allow_pickle=True)

def adjust_bbox(bbox, ih, iw, padding=0.2):
    x, y, w, h = bbox
    x -= padding * w
    y -= padding * h
    w += 2 * padding * w
    h += 2 * padding * h

    # Ensure the coordinates are within bounds
    x = max(int(x), 0)
    y = max(int(y), 0)
    w = min(int(x + w), iw)
    h = min(int(y + h), ih)
    return x, y, w, h

def identify_face(image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract face embeddings from the uploaded image
    new_face_encodings = face_recognition.face_encodings(image_rgb)
    
    if len(new_face_encodings) == 0:
        return "No face detected in the image."
    
    # Iterate over detected faces (in case there are multiple)
    for new_face_encoding in new_face_encodings:
        # Initialize the best match variables
        best_match_name = "Unknown"
        best_match_distance = float('inf')
        
        # Compare against known embeddings
        for person_name, embeddings_list in face_embeddings.items():
            distances = distance.cdist([new_face_encoding], embeddings_list, "euclidean")
            min_distance = np.min(distances)

            # Find the best match based on the minimum distance
            if min_distance < best_match_distance and min_distance < 0.6:  # 0.6 is a typical threshold
                best_match_distance = min_distance
                best_match_name = person_name
        
        # Return the best match found
        return best_match_name
    
    return "Unknown face"

# Initialize PiCamera
picam2 = Picamera2()
picam2.start_preview()
picam2.start()

def make_call(phone_number):
    """
    Place a call by executing the call.py script.
    """
    command = f'sudo python3 call.py {phone_number}'
    os.system(command)

try:
    while True:
        # Capture frame from PiCamera
        frame = picam2.capture_array()

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Perform face detection
        results = face_detection.process(frame_rgb)

        # Draw bounding boxes and identify faces
        if results.detections:
            ih, iw, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bbox = adjust_bbox(bbox, ih, iw, padding=0.2)

                # Extract face region
                face_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # Identify face
                result = identify_face(face_image)
                print(f"Identified as: {result}")

                # Draw rectangle
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

                # Make a call if an unknown face is detected
                if result == "Unknown":
                    print("Unknown face detected. Making a call...")
                    phone_number = '0709735982'  # Adjust this to your number
                    make_call(phone_number)
                    time.sleep(10)  # Wait for 10 seconds to avoid repeated calls

        # Display the resulting frame
        cv2.imshow('Face Detection and Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    picam2.stop()
    cv2.destroyAllWindows()
