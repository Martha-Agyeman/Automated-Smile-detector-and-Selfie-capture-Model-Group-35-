from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the smile detection model
model = load_model('Final_model.h5')

# Initialize the camera
camera = cv2.VideoCapture(0)

def predict_smile(frame):
    # Resize the frame for faster processing
    frame = cv2.resize(frame, (256, 256))

    # Preprocess the frame for smile detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (256, 256))

    roi = roi.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # Predict smile using the model
    prediction = model.predict(roi)[0]
    is_smiling = prediction > 0.5  # Use a threshold for binary classification

    # Draw a rectangle and label on the frame
    label = 'Smiling' if is_smiling else 'Not Smiling'
    color = (0, 255, 0) if is_smiling else (0, 0, 255)
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (10, 60), (630, 420), color, 2)

    # Encode the frame to JPEG format
    ret, processed_frame = cv2.imencode('.jpg', frame)

    return is_smiling, processed_frame



def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            is_smiling, processed_frame = predict_smile(frame)

            # Draw a rectangle and label on the frame
            label = 'Smiling' if is_smiling else 'Not Smiling'
            color = (0, 255, 0) if is_smiling else (0, 0, 255)
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (10, 60), (630, 420), color, 2)

            # Encode the frame to JPEG format
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            # Yield the frame for HTML rendering
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + processed_frame.tobytes() + b'\r\n\r\n')    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    success, frame = camera.read()

    if success:
        cv2.imwrite('captured_photo.jpg', frame)
        return "Photo captured successfully!"

    return "Failed to capture photo."

if __name__ == "__main__":
    app.run(debug=True)
