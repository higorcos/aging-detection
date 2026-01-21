import os
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 96

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "idade_model.keras")
##IMAGE_PATH = os.path.join(BASE_DIR, "..", "imagens_teste", "rosto17.png")
IMAGE_PATH = os.path.join(BASE_DIR, "..", "imagens_teste", "pai2.png")

model = tf.keras.models.load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError("Imagem não encontrada")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ========================
# IDADE (IMAGEM INTEIRA)
# ========================
img_full = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
img_full = img_full / 255.0
img_full = img_full.reshape(1, IMG_SIZE, IMG_SIZE, 1)

pred = model.predict(img_full)
idade = int(pred[0][0])

# ========================
# DETECÇÃO DE ROSTO (PDI)
# ========================
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    rosto = gray[y:y+h, x:x+w]

    edges = cv2.Canny(rosto, 100, 200)
    rugas_score = np.sum(edges) / (w * h)

    if rugas_score > 15:
        envelhecimento = "Sinais evidentes"
        cor = (0, 0, 255)
    else:
        envelhecimento = "Poucos sinais"
        cor = (0, 255, 0)

    cv2.rectangle(image, (x, y), (x+w, y+h), cor, 2)

    cv2.putText(image, f"Idade estimada: {idade} anos",
                (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

    cv2.putText(image, envelhecimento,
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

cv2.imshow("Analise Facial", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
