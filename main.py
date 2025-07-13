import cv2
from ultralytics import YOLO
from deepface import DeepFace

# Load YOLOv8 face detection model
model = YOLO("yolov8n-face-lindevs.pt")  # or yolov8n.pt and filter person

# Define function to display ad
def show_ad(gender, age_bucket):
    print(f"Showing ad for {gender}, Age: {age_bucket}")
    if gender == 'Male':
        ad_file = 'ads/men_young.jpg' if age_bucket in ['(15-20)', '(25-32)'] else 'ads/men_mature.jpg'
    else:
        ad_file = 'ads/women_young.jpg' if age_bucket in ['(15-20)', '(25-32)'] else 'ads/women_mature.jpg'

    ad_img = cv2.imread(ad_file)
    if ad_img is not None:
        cv2.imshow("Advertisement", ad_img)
        cv2.waitKey(5000)
        cv2.destroyWindow("Advertisement")
    else:
        print("Ad not found:", ad_file)

# Use DeepFace to detect age and gender
def predict_age_gender_pytorch(face_img):
    analysis = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False)
    age = analysis[0]['age']
    gender = analysis[0]['gender']
    gender = 'Male' if gender == 'Man' else 'Female'

    # Bucket the age
    if age < 3:
        age_bucket = "(0-2)"
    elif age < 7:
        age_bucket = "(4-6)"
    elif age < 13:
        age_bucket = "(8-12)"
    elif age < 21:
        age_bucket = "(15-20)"
    elif age < 35:
        age_bucket = "(25-32)"
    elif age < 45:
        age_bucket = "(38-43)"
    elif age < 55:
        age_bucket = "(48-53)"
    else:
        age_bucket = "(60-100)"
        
    return gender, age_bucket

# Webcam stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # Predict age and gender with PyTorch model
            gender, age_bucket = predict_age_gender_pytorch(face_img)

            label = f"{gender}, {age_bucket}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            show_ad(gender, age_bucket)
            break  # show one ad per frame

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


'''import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Load your Keras models
age_model = tf.keras.models.load_model('age_model.h5')
gender_model = tf.keras.models.load_model('gender_model.h5')

# Load YOLOv8 face detection model
face_detector = YOLO("yolov8n-face-lindevs.pt")

# Age bucketing
def get_age_bucket(age):
    if age < 3:
        return "(0-2)"
    elif age < 7:
        return "(4-6)"
    elif age < 13:
        return "(8-12)"
    elif age < 21:
        return "(15-20)"
    elif age < 35:
        return "(25-32)"
    elif age < 45:
        return "(38-43)"
    elif age < 55:
        return "(48-53)"
    else:
        return "(60-100)"

# Display ad based on prediction
def show_ad(gender, age_bucket):
    print(f"Showing ad for {gender}, Age Bucket: {age_bucket}")
    if gender == 'Male':
        ad_path = 'ads/men_young.jpg' if age_bucket in ['(15-20)', '(25-32)'] else 'ads/men_mature.jpg'
    else:
        ad_path = 'ads/women_young.jpg' if age_bucket in ['(15-20)', '(25-32)'] else 'ads/women_mature.jpg'
    
    ad_img = cv2.imread(ad_path)
    if ad_img is not None:
        cv2.imshow("Advertisement", ad_img)
        cv2.waitKey(5000)
        cv2.destroyWindow("Advertisement")
    else:
        print("Ad image not found:", ad_path)

# Preprocess face for Keras models
def preprocess_face_for_keras(face_img):
    resized = cv2.resize(face_img, (224, 224))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Predict age and gender
def predict_age_gender(face_img):
    face_input = preprocess_face_for_keras(face_img)
    predicted_age = age_model.predict(face_input)[0][0]
    gender_prob = gender_model.predict(face_input)[0][0]
    predicted_gender = 'Male' if gender_prob < 0.5 else 'Female'
    age_bucket = get_age_bucket(predicted_age)
    return predicted_gender, age_bucket, predicted_age

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection using YOLO
    results = face_detector(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # Predict age and gender
            gender, age_bucket, raw_age = predict_age_gender(face_img)

            # Draw rectangle and label
            label = f"{gender}, {int(raw_age)}y"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show ad (only once per detected face)
            show_ad(gender, age_bucket)
            break  # Show ad for first face only

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
