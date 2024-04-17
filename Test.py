import cv2 as cv
import tensorflow as tf
from Model import Model
from deepface import DeepFace
import numpy as np

# Load the model
modelObj = Model()
model = modelObj.model

# Load weights for the model
model.load_weights("Weights/best_weights_checkpoint.h5")

camera = cv.VideoCapture(0)

initial_batch_size = 5  # Set an initial batch size
batch_size = initial_batch_size

while cv.waitKey(1) & 0xFF != ord("q"):
    _, img = camera.read()
    try:
        # Extract faces from the modified image
        extracted_face = DeepFace.extract_faces(
            img,
            target_size=(224, 224),
            enforce_detection=True,
            detector_backend='mediapipe'
        )[0]
        faceRegions = []
        x, y, w, h, _, _  = extracted_face["facial_area"].values()
        face = extracted_face["face"].astype('uint8')
        faceRegions.append(face)

        faceRegionsTensor = tf.convert_to_tensor(faceRegions, dtype=tf.float64)

        try:
            # Model prediction
            mask, binary = model(faceRegionsTensor)
        except tf.errors.ResourceExhaustedError:
            print("Reducing batch size due to memory constraints")
            batch_size = max(1, batch_size // 2)  # Reduce batch size

        res = tf.reduce_mean(mask).numpy()
        # res = binary.item()

        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        print(res)
        if  res > 0.1:
            cv.putText(
                img, "Real", (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0)
            )
            
        else:
            cv.putText(
                img, "Fake", (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255)
            )
    except Exception as e:
        print(e)

    cv.imshow("Camera", img)