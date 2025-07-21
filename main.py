# # import cv2
# # from ultralytics import YOLO
# # from deepface import DeepFace

# # # Load YOLOv8 face detection model
# # model = YOLO("yolov8n-face-lindevs.pt")  # or yolov8n.pt and filter person

# # # Define function to display ad
# # def show_ad(gender, age_bucket):
# #     print(f"Showing ad for {gender}, Age: {age_bucket}")
# #     if gender == 'Male':
# #         ad_file = 'ads/men_young.jpg' if age_bucket in ['(15-20)', '(25-32)'] else 'ads/men_mature.jpg'
# #     else:
# #         ad_file = 'ads/women_young.jpg' if age_bucket in ['(15-20)', '(25-32)'] else 'ads/women_mature.jpg'

# #     ad_img = cv2.imread(ad_file)
# #     if ad_img is not None:
# #         cv2.imshow("Advertisement", ad_img)
# #         cv2.waitKey(5000)
# #         cv2.destroyWindow("Advertisement")
# #     else:
# #         print("Ad not found:", ad_file)

# # # Use DeepFace to detect age and gender
# # def predict_age_gender_pytorch(face_img):
# #     analysis = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False)
# #     age = analysis[0]['age']
# #     gender = analysis[0]['gender']
# #     gender = 'Male' if gender == 'Male' else 'Female'

# #     # Bucket the age
# #     if age < 3:
# #         age_bucket = "(0-2)"
# #     elif age < 7:
# #         age_bucket = "(4-6)"
# #     elif age < 13:
# #         age_bucket = "(8-12)"
# #     elif age < 21:
# #         age_bucket = "(15-20)"
# #     elif age < 35:
# #         age_bucket = "(25-32)"
# #     elif age < 45:
# #         age_bucket = "(38-43)"
# #     elif age < 55:
# #         age_bucket = "(48-53)"
# #     else:
# #         age_bucket = "(60-100)"
        
# #     return gender, age_bucket

# # # Webcam stream
# # cap = cv2.VideoCapture(0)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     results = model(frame)
# #     for result in results:
# #         for box in result.boxes:
# #             x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
# #             face_img = frame[y1:y2, x1:x2]
# #             if face_img.size == 0:
# #                 continue

# #             # Predict age and gender with PyTorch model
# #             gender, age_bucket = predict_age_gender_pytorch(face_img)

# #             label = f"{gender}, {age_bucket}"
# #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
# #             cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

# #             show_ad(gender, age_bucket)
# #             break  # show one ad per frame

# #     cv2.imshow("Camera Feed", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()


# # '''import cv2
# # import numpy as np
# # import tensorflow as tf
# # from ultralytics import YOLO

# # # Load your Keras models
# # age_model = tf.keras.models.load_model('age_model.h5')
# # gender_model = tf.keras.models.load_model('gender_model.h5')

# # # Load YOLOv8 face detection model
# # face_detector = YOLO("yolov8n-face-lindevs.pt")

# # # Age bucketing
# # def get_age_bucket(age):
# #     if age < 3:
# #         return "(0-2)"
# #     elif age < 7:
# #         return "(4-6)"
# #     elif age < 13:
# #         return "(8-12)"
# #     elif age < 21:
# #         return "(15-20)"
# #     elif age < 35:
# #         return "(25-32)"
# #     elif age < 45:
# #         return "(38-43)"
# #     elif age < 55:
# #         return "(48-53)"
# #     else:
# #         return "(60-100)"

# # # Display ad based on prediction
# # def show_ad(gender, age_bucket):
# #     print(f"Showing ad for {gender}, Age Bucket: {age_bucket}")
# #     if gender == 'Male':
# #         ad_path = 'ads/men_young.jpg' if age_bucket in ['(15-20)', '(25-32)'] else 'ads/men_mature.jpg'
# #     else:
# #         ad_path = 'ads/women_young.jpg' if age_bucket in ['(15-20)', '(25-32)'] else 'ads/women_mature.jpg'
    
# #     ad_img = cv2.imread(ad_path)
# #     if ad_img is not None:
# #         cv2.imshow("Advertisement", ad_img)
# #         cv2.waitKey(5000)
# #         cv2.destroyWindow("Advertisement")
# #     else:
# #         print("Ad image not found:", ad_path)

# # # Preprocess face for Keras models
# # def preprocess_face_for_keras(face_img):
# #     resized = cv2.resize(face_img, (224, 224))
# #     normalized = resized / 255.0
# #     return np.expand_dims(normalized, axis=0)

# # # Predict age and gender
# # def predict_age_gender(face_img):
# #     face_input = preprocess_face_for_keras(face_img)
# #     predicted_age = age_model.predict(face_input)[0][0]
# #     gender_prob = gender_model.predict(face_input)[0][0]
# #     predicted_gender = 'Male' if gender_prob < 0.5 else 'Female'
# #     age_bucket = get_age_bucket(predicted_age)
# #     return predicted_gender, age_bucket, predicted_age

# # # Start webcam
# # cap = cv2.VideoCapture(0)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     # Face detection using YOLO
# #     results = face_detector(frame)
# #     for result in results:
# #         for box in result.boxes:
# #             x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
# #             face_img = frame[y1:y2, x1:x2]
# #             if face_img.size == 0:
# #                 continue

# #             # Predict age and gender
# #             gender, age_bucket, raw_age = predict_age_gender(face_img)

# #             # Draw rectangle and label
# #             label = f"{gender}, {int(raw_age)}y"
# #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             cv2.putText(frame, label, (x1, y1 - 10),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# #             # Show ad (only once per detected face)
# #             show_ad(gender, age_bucket)
# #             break  # Show ad for first face only

# #     cv2.imshow("Camera Feed", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()
# # '''


# import cv2
# import json
# from fastapi import FastAPI
# from pydantic import BaseModel
# from ultralytics import YOLO
# from deepface import DeepFace
# from fastapi.responses import JSONResponse
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware


# app = FastAPI()

# # Allow all origins for testing (you can restrict later)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],   # Allow all for now
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )




# # Load models
# yolo_model = YOLO("yolov8n-face-lindevs.pt")

# # Age bucketing
# def bucket_age(age: int):
#     if age < 3:
#         return "(0-2)"
#     elif age < 7:
#         return "(4-6)"
#     elif age < 13:
#         return "(8-12)"
#     elif age < 21:
#         return "(15-20)"
#     elif age < 35:
#         return "(25-32)"
#     elif age < 45:
#         return "(38-43)"
#     elif age < 55:
#         return "(48-53)"
#     else:
#         return "(60-100)"

# # Load ad metadata from JSON
# def load_ads():
#     with open("ads.json", "r") as f:
#         return json.load(f)

# # Process camera and predict gender/age
# def detect_person_details():
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     cap.release()

#     if not ret:
#         return None

#     results = yolo_model(frame)
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             face_crop = frame[y1:y2, x1:x2]
#             if face_crop.size == 0:
#                 continue
#             try:
#                 analysis = DeepFace.analyze(face_crop, actions=["age", "gender"], enforce_detection=False)
#                 age = analysis[0]["age"]
#                 gender = "Male" if analysis[0]["gender"] == "Man" else "Female"
#                 age_group = bucket_age(age)
#                 return {"gender": gender, "age_group": age_group}
#             except:
#                 return None
#     return None



# class PersonResponse(BaseModel):
#     gender: str
#     age_group: str
#     matched_ads: list

# # @app.get("/detect_person", response_model=PersonResponse)
# # def detect_and_get_ads():
# #     person = detect_person_details()
# #     if not person:
# #         return JSONResponse(status_code=404, content={"message": "No face detected"})

# #     ads_data = load_ads()
# #     filtered_ads = [ad for ad in ads_data if ad["gender"] == person["gender"] and ad["age_group"] == person["age_group"]]

# #     return {
# #         "gender": person["gender"],
# #         "age_group": person["age_group"],
# #         "matched_ads": filtered_ads
# #     }


# @app.get("/detect_person", response_model=PersonResponse)
# def detect_and_get_ads():
#     person = detect_person_details()
#     if not person:
#         print("❌ No face detected.")
#         return JSONResponse(status_code=404, content={"message": "No face detected"})

#     print(f"✅ Detected Person → Gender: {person['gender']}, Age Group: {person['age_group']}")

#     ads_data = load_ads()
#     filtered_ads = [
#         ad for ad in ads_data
#         if ad["gender"] == person["gender"] and ad["age_group"] == person["age_group"]
#     ]

#     if not filtered_ads:
#         print("⚠ No matching ads found.")
#     else:
#         print(f"🎯 {len(filtered_ads)} matching ad(s) found:")
#         for ad in filtered_ads:
#             print(f"  - Ad Image: {ad['ad_image']}")
#             print(f"    QR Code: {ad['qr_code']}")

#     return {
#         "gender": person["gender"],
#         "age_group": person["age_group"],
#         "matched_ads": filtered_ads
#     }


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from deepface import DeepFace
import cv2
import json

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve ads folder
from fastapi.staticfiles import StaticFiles
app.mount("/ads", StaticFiles(directory="ads"), name="ads")

yolo_model = YOLO("yolov8n-face-lindevs.pt")

def bucket_age(age: int):
    if age < 3: return "(0-2)"
    elif age < 7: return "(4-6)"
    elif age < 13: return "(8-12)"
    elif age < 21: return "(15-20)"
    elif age < 35: return "(25-32)"
    elif age < 45: return "(38-43)"
    elif age < 55: return "(48-53)"
    else: return "(60-100)"

def load_ads():
    with open("ads.json", "r") as f:
        return json.load(f)

def detect_person_details():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    results = yolo_model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            try:
                analysis = DeepFace.analyze(face_crop, actions=["age", "gender"], enforce_detection=False)
                age = analysis[0]["age"]
                gender = "Male" if analysis[0]["gender"] == "Man" else "Female"
                return {"gender": gender, "age_group": bucket_age(age)}
            except:
                return None
    return None

class PersonResponse(BaseModel):
    gender: str
    age_group: str
    matched_ads: list

@app.get("/detect_person", response_model=PersonResponse)
def detect_and_get_ads():
    person = detect_person_details()
    if not person:
        return JSONResponse(status_code=404, content={"message": "No face detected"})
    ads_data = load_ads()
    filtered_ads = [
        ad for ad in ads_data
        if ad["gender"] == person["gender"] and ad["age_group"] == person["age_group"]
    ]
    return {
        "gender": person["gender"],
        "age_group": person["age_group"],
        "matched_ads": filtered_ads
    }
