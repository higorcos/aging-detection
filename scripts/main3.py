import os
import cv2
import numpy as np
import tensorflow as tf

# ========================
# CONFIGURAÇÕES
# ========================
IMG_SIZE = 96

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "idade_model3.keras")
#IMAGE_PATH = os.path.join(BASE_DIR, "..", "imagens_teste", "rosto.png")
#IMAGE_PATH = os.path.join(BASE_DIR, "..", "imagens_teste", "mae2.png")
IMAGE_PATH = os.path.join(BASE_DIR, "..", "imagens_teste", "rosto.png")

# ========================
# CARREGAR MODELO
# ========================
model = tf.keras.models.load_model(MODEL_PATH)

# ========================
# DETECTOR DE ROSTO
# ========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ========================
# FUNÇÃO DE ESTÁGIO
# ========================
def classificar_envelhecimento(idade):
    if idade <= 12:
        return "Crianca"
    elif idade <= 25:
        return "Jovem"
    elif idade <= 39:
        return "Adulto Jovem"
    elif idade <= 59:
        return "Adulto"
    else:
        return "Idoso"

# ========================
# CARREGAR IMAGEM
# ========================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Imagem não encontrada")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# ========================
# PROCESSAR ROSTOS
# ========================
for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0
    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    pred = model.predict(face)
    idade = int(pred[0][0])

    estagio = classificar_envelhecimento(idade)

    # ========================
    # DESENHO NA IMAGEM
    # ========================
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    texto1 = f"Idade: {idade} anos"
    texto2 = f"Estagio: {estagio}"

    cv2.putText(img, texto1, (x, y-25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.putText(img, texto2, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

# ========================
# EXIBIR RESULTADO
# ========================
cv2.imshow("Analise de Envelhecimento Facial", img)
cv2.waitKey(0)
cv2.destroyAllWindows() 

""" window_name="Analise de Envelhecimento Facial"

# Redimensionar a imagem para caber na tela, se necessário
img_resized = cv2.resize(img, (600, 800))
cv2.imshow(window_name, img_resized)

cv2.waitKey(0)
cv2.destroyAllWindows() """