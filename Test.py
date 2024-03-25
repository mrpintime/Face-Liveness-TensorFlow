import cv2 as cv
import tensorflow as tf
from Model import Model

# Load the model
modelObj = Model()
model = modelObj.model

# Load weights for the model
model.load_weights("Weights/best_weights_checkpoint.h5")


# Transformation pipeline
def preprocess_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    img = img / 255.0  # Normalize to [0,1]
    return img


faceClassifier = cv.CascadeClassifier("Classifiers/haarface.xml")
camera = cv.VideoCapture(0)

initial_batch_size = 5  # Set an initial batch size
batch_size = initial_batch_size

while cv.waitKey(1) & 0xFF != ord("q"):
    _, img = camera.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)

    # Process faces with dynamic batch size handling
    for i in range(0, len(faces), batch_size):
        faceRegions = []
        for j in range(i, min(i + batch_size, len(faces))):
            x, y, w, h = faces[j]
            faceRegion = img[y : y + h, x : x + w]
            faceRegion = preprocess_image(faceRegion)
            faceRegions.append(faceRegion)

        faceRegionsTensor = tf.convert_to_tensor(faceRegions, dtype=tf.float32)

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
        if res < 0.5:
            cv.putText(
                img, "Fake", (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255)
            )
        else:
            cv.putText(
                img, "Real", (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0)
            )

    cv.imshow("Camera", img)
