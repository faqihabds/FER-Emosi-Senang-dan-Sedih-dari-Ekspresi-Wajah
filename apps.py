import os
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Dictionary for class labels
dic = {0: 'happy', 1: 'sad'}


# # Load the FER model
from keras.config import enable_unsafe_deserialization
enable_unsafe_deserialization()
model = tf.keras.models.load_model('inception-resnet_2c.keras')
# model = load_model('inception-resnet_2c.keras')

# Ensure the model prediction function is ready
model.make_predict_function()

def predict_label(img_path):
    """Function to predict the label of the input image using bounding box."""
    # Load the image in grayscale
    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Load the original image (for drawing bounding box)
    original_image = cv2.imread(img_path)

    # Detect face using a Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # Ensure at least one face is detected
    if len(faces) == 0:
        return "No face detected", img_path

    # Process each detected face
    for (x, y, w, h) in faces:
        # Crop the detected face
        cropped_face = gray_image[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (48, 48))
        cropped_face = np.expand_dims(cropped_face, axis=-1)  # Add channel dimension
        cropped_face = cropped_face / 255.0  # Normalize the image
        cropped_face = np.expand_dims(cropped_face, axis=0)  # Add batch dimension

        # Predict the class
        probabilities = model.predict(cropped_face)[0]
        max_index = np.argmax(probabilities)
        label = dic[max_index]
        confidence = probabilities[max_index] * 100  # Convert to percentage

        # Draw bounding box and label with percentage
        cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box
        label_text = f"{label}: {confidence:.2f}%"
        cv2.putText(original_image, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the image with bounding boxes and labels
    boxed_img_path = img_path.replace("static/", "static/boxed_")
    cv2.imwrite(boxed_img_path, original_image)

    return "Processed successfully", boxed_img_path

def process_video(video_path): 
    """Function to process video frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Cannot open video file"

    # Output video settings
    output_path = video_path.replace("static/", "static/processed_")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Face cascade for detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop when no more frames are available

            frame_count += 1

            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Crop and preprocess the detected face
                cropped_face = gray_frame[y:y+h, x:x+w]
                cropped_face = cv2.resize(cropped_face, (48, 48))
                cropped_face = np.expand_dims(cropped_face, axis=-1)  # Add channel dimension
                cropped_face = cropped_face / 255.0  # Normalize the image
                cropped_face = np.expand_dims(cropped_face, axis=0)  # Add batch dimension

                # Predict the class
                p = model.predict(cropped_face)
                label = dic[np.argmax(p[0])]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label_text = f"{label}: {p[0][np.argmax(p[0])]*100:.2f}%"
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Write the processed frame to the output video
            out.write(frame)

        except cv2.error as e:
            print(f"OpenCV error at frame {frame_count}: {e}")
            continue
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue

        # Debug: Print progress
        print(f"Processing frame {frame_count}/{total_frames}")

    # Release resources
    cap.release()
    out.release()
    return output_path


# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    """Render the main index page."""
    return render_template("index.html")

# @app.route("/submit", methods=['GET', 'POST'])
# def get_output():
#     """Handle the image upload and return prediction results."""
#     if request.method == 'POST':
#         img = request.files['my_image']

#         # Save the uploaded image
#         img_path = os.path.join("static", img.filename)
#         img.save(img_path)

#         # Get prediction
#         p = predict_label(img_path)

#         # Render the results in the template
#         return render_template("index.html", prediction=p, img_path=img_path)

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    """Handle the image or video upload and return prediction results."""
    if request.method == 'POST':
        file = request.files['my_file']
        file_path = os.path.join("static", file.filename)
        file.save(file_path)

        # Check if the file is an image or video
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Process image
            p, boxed_img_path = predict_label(file_path)
            return render_template("index.html", prediction=p, img_path=boxed_img_path)

        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Process video
            processed_video_path = process_video(file_path)
            return render_template("index.html", video_path=processed_video_path)

        else:
            return "Unsupported file type"



if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
