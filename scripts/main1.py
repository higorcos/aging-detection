#funciona mais com erro

import cv2
import numpy as np
from tensorflow.keras.models import load_model

##model = load_model("models/cnn_envelhecimento_utkface.h5")
model = load_model("models/cnn_envelhecimento_melhorado.h5")

classes = ["Jovem", "Envelhecimento Moderado", "Envelhecimento Avançado"]
def preprocessar_rosto(imagem_path):
    imagem = cv2.imread(imagem_path)
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    rosto = gray[y:y+h, x:x+w]

    rosto = cv2.resize(rosto, (128, 128))
    rosto = rosto / 255.0
    rosto = np.reshape(rosto, (1, 128, 128, 1))

    return rosto, (x, y, w, h)


imagem_path = "imagens_teste/rosto2.png"


rosto_input, bbox = preprocessar_rosto(imagem_path)

if rosto_input is None:
    print("Nenhum rosto detectado.")
    exit()

pred = model.predict(rosto_input)
classe = classes[np.argmax(pred)]

print("Resultado:", classe)

imagem = cv2.imread(imagem_path)

x, y, w, h = bbox
cv2.rectangle(imagem, (x, y), (x+w, y+h), (0,255,0), 2)
cv2.putText(
    imagem, classe, (x, y-10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2
)

cv2.imshow("Resultado", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

