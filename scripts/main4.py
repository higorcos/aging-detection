import cv2
import numpy as np
import tensorflow as tf
import os

# =========================
# CONFIGURAÇÕES
# =========================
IMG_SIZE = 96

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "idade_model3.keras")
IMAGE_PATH = os.path.join(BASE_DIR, "..", "imagens_teste", "rosto.png")

# =========================
# MODELO
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# DETECTOR DE ROSTO
# =========================
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
        recs.append("Retinol à noite")
        recs.append("Protetor solar FPS 50")

    if manchas > 15:
        recs.append("Vitamina C pela manhã")
        recs.append("Niacinamida")

    if tipo_pele == "Seca":
        recs.append("Hidratante com ácido hialurônico")

    if tipo_pele == "Oleosa":
        recs.append("Gel oil-free")
        recs.append("Limpeza 2x ao dia")

    if not recs:
        recs.append("Manter rotina atual")

    return recs

# =========================
# LEITURA DA IMAGEM
# =========================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Imagem não encontrada")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# DETECÇÃO DO ROSTO
# =========================
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    raise ValueError("Nenhum rosto detectado")

x, y, w, h = faces[0]
rosto = img[y:y+h, x:x+w]
rosto_gray = gray[y:y+h, x:x+w]

# =========================
# PREPROCESSAMENTO
# =========================
face_resized = cv2.resize(rosto_gray, (IMG_SIZE, IMG_SIZE))
face_norm = face_resized / 255.0
face_norm = face_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# =========================
# PREDIÇÃO DE IDADE
# =========================
idade = int(model.predict(face_norm)[0][0])

# =========================
# ANÁLISES DE PELE
# =========================
rugas_score = analisar_rugas(rosto_gray)
manchas_score = analisar_manchas(rosto)
tipo_pele = analisar_oleosidade(rosto_gray)

recomendacoes = gerar_recomendacoes(
    rugas_score, manchas_score, tipo_pele
)

# =========================
# VISUALIZAÇÃO
# =========================
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

info = [
    f"Idade estimada: {idade} anos",
    f"Tipo de pele: {tipo_pele}",
    f"Rugas score: {rugas_score:.1f}",
    f"Manchas score: {manchas_score:.1f}",
]

y_text = y - 10
for text in info:
    cv2.putText(
        img, text, (x, y_text),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 255, 0), 2
    )
    y_text -= 20

# Recomendações
y_rec = y + h + 20
cv2.putText(
    img, "Cuidados recomendados:",
    (x, y_rec),
    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
    (255, 255, 255), 2
)

for rec in recomendacoes:
    y_rec += 25
    cv2.putText(
        img, f"- {rec}",
        (x, y_rec),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1
    )

# =========================
# JANELA CENTRALIZADA
# =========================
h_img, w_img = img.shape[:2]
cv2.namedWindow("Analise de Envelhecimento Facial", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Analise de Envelhecimento Facial", 900, 700)
cv2.moveWindow(
    "Analise de Envelhecimento Facial",
    (1920 - 900) // 2,
    (1080 - 700) // 2
)

cv2.imshow("Analise de Envelhecimento Facial", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
