#pip install opencv-python

import cv2
import os

# Function to create a directory to save captured images
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Setup for capturing video from the default camera
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory to save the images
home = os.path.expanduser("~")  # gets the home directory
save_folder = os.path.join(home, "Desktop", "face_data/fd/")

#for windows
#home = os.path.expanduser("~")  # gets the home directory
#save_folder = os.path.join(home, "Desktop", "captured_faces")

ensure_directory(save_folder)

# Count to name the saved image files
img_count = 0

try:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale for the cascade classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(save_folder, f"face_{img_count}.jpg"), face_img)
            img_count += 1

        # Display the frame
        cv2.imshow('Capturing Faces', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()