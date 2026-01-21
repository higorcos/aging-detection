from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
import os
import uuid

app = Flask(__name__)

# =========================
# CONFIGURAÇÕES
# =========================
IMG_SIZE = 96
UPLOAD_FOLDER = "web/static/uploads"
RESULT_FOLDER = "web/static/results"
RESULT_FOLDER = "web/static/results"
RESULT_FOLDER1 = "static/results"
 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = tf.keras.models.load_model("models/idade_model.keras")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# FUNÇÕES DE ANÁLISE
# =========================
def analisar_rugas(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return np.mean(np.abs(lap))

def analisar_manchas(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    return np.std(l)

def analisar_oleosidade(gray):
    brilho = np.mean(gray)
    if brilho < 90:
        return "Seca"
    elif brilho > 160:
        return "Oleosa"
    else:
        return "Normal"

def gerar_recomendacoes(rugas, manchas, tipo_pele):
    recs = []
    if rugas > 10:
        recs += ["Retinol à noite", "Protetor solar FPS 50"]
    if manchas > 15:
        recs += ["Vitamina C", "Niacinamida"]
    if tipo_pele == "Seca":
        recs.append("Hidratante com ácido hialurônico")
    if tipo_pele == "Oleosa":
        recs += ["Gel oil-free", "Limpeza 2x ao dia"]
    if not recs:
        recs.append("Manter rotina atual")
    return recs

# =========================
# ROTAS
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"

    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "Nenhum rosto detectado"

    x, y, w, h = faces[0]
    rosto = img[y:y+h, x:x+w]
    rosto_gray = gray[y:y+h, x:x+w]

    # IA
    face = cv2.resize(rosto_gray, (IMG_SIZE, IMG_SIZE)) / 255.0
    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    idade = int(model.predict(face)[0][0])

    # Análises
    rugas = analisar_rugas(rosto_gray)
    manchas = analisar_manchas(rosto)
    tipo = analisar_oleosidade(rosto_gray)
    recs = gerar_recomendacoes(rugas, manchas, tipo)

    # Desenhar
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(img, f"Idade: {idade}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    result_path = os.path.join(RESULT_FOLDER, filename)
    result_path1 = os.path.join(RESULT_FOLDER1, filename)
    cv2.imwrite(result_path, img)
 
    return render_template(
        "result.html",
        idade=idade,
        tipo=tipo,
        recs=recs,
        image_path=result_path1
    )

if __name__ == "__main__":
    app.run(debug=True)
