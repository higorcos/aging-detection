import os
import uuid
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for

# ========================
# CONFIGURAÇÕES GERAIS
# ========================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "index")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "idade_model3.keras")

IMG_SIZE = 96

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

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
# FUNÇÕES AUXILIARES
# ========================
def preprocess_face(face):
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


def analisar_envelhecimento(idade):
    if idade < 25:
        return "Pele jovem. Manter hidratação e proteção solar diária."
    elif idade < 40:
        return "Início do envelhecimento. Usar antioxidantes e protetor solar."
    elif idade < 55:
        return "Envelhecimento moderado. Foco em colágeno e hidratação intensa."
    else:
        return "Envelhecimento avançado. Cuidados dermatológicos contínuos."


# ========================
# ROTAS
# ========================
@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/analisar", methods=["POST"])
def analisar():
    if "foto" not in request.files:
        return redirect(url_for("index"))

    file = request.files["foto"]

    if file.filename == "":
        return redirect(url_for("index"))

    # Nome único
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    # Ler imagem
    img = cv2.imread(upload_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return render_template(
            "resultado.html",
            erro="Nenhum rosto detectado.",
            imagem=None
        )

    # Pega o maior rosto
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y+h, x:x+w]

    face_inp_

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
