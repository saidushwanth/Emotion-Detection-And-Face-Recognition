 Emotion Detection and Face Recognition for Attendance System
This project combines real-time emotion detection and face recognition to build an intelligent attendance system using deep learning and web technology.

🔍 Features
✅ Real-time face detection using OpenCV

😄 Emotion classification into categories like Happy, Sad, Angry, etc.

🧠 Deep learning model using CNN for emotion recognition

🧾 Automated attendance system using facial recognition

🌐 Web interface for interaction and reporting

🛠️ Tech Stack
Frontend: HTML, CSS, JavaScript

Backend: Python (Flask)

Libraries/Tools:

OpenCV

TensorFlow/Keras

NumPy, Pandas

face_recognition

Matplotlib

SQLite (or CSV for attendance)

🖼️ Emotion Categories
The model is trained to recognize the following emotions:

Angry 😠

Disgust 🤢

Fear 😨

Happy 😄

Sad 😢

Surprise 😲

Neutral 😐

📁 Project Structure
php
Copy
Edit
Emotion-Detection-And-Face-Recognition/
│
├── dataset/                  # Preprocessed image data
├── models/                   # Saved CNN model and face encodings
├── static/                   # CSS, JS, images
├── templates/                # HTML templates
├── attendance.csv            # Log file for attendance
├── app.py                    # Main Flask application
├── train_model.py            # Script to train emotion model
├── face_register.py          # Script to register new faces
├── README.md                 # Project description
🚀 How to Run
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/saidushwanth/Emotion-Detection-And-Face-Recognition.git
cd Emotion-Detection-And-Face-Recognition
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Train the emotion recognition model
bash
Copy
Edit
python train_model.py
4. Register faces for attendance
bash
Copy
Edit
python face_register.py
5. Run the app
bash
Copy
Edit
python app.py
6. Access the app
Open your browser and go to http://127.0.0.1:5000/

📊 Sample Output
Live webcam detecting face and emotion

Emotion label overlay

Name tag and auto-marked attendance

Attendance saved in CSV file

📚 Future Improvements
Add database integration for large-scale deployment

Improve model accuracy with larger datasets

Add speaker emotion detection and audio support

🙌 Acknowledgements
FER2013 Dataset

OpenCV, face_recognition, Keras
