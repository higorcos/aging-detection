import os
import cv2

# ========================
# CONFIGURAÇÕES
# ========================
INPUT_DIR = "dataset/archive1"
OUTPUT_DIR = "dataset/dataset_padronizado"
IMG_SIZE = 96

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# DETECTOR DE ROSTO
# ========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extrair_idade(nome):
    try:
        parte = nome.split("_")[1]      # 1F37.JPG
        idade = int(parte[2:4])         # pega só "37"
        return idade
    except:
        return None


# ========================
# PROCESSAMENTO
# ========================
idx = 0

for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if not file.lower().endswith(".jpg"):
            continue

        idade = extrair_idade(file)
        if idade is None:
            continue

        img_path = os.path.join(root, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
        if len(faces) == 0:
            continue

        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        if w < 30 or h < 30:
            continue

        rosto = gray[y:y+h, x:x+w]
        rosto = cv2.resize(rosto, (IMG_SIZE, IMG_SIZE))

        nome_saida = f"{idade}_{idx}.jpg"
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, nome_saida),
            rosto
        )

        idx += 1

        if idx % 500 == 0:
            print(f"{idx} imagens processadas")

print("✅ Dataset padronizado com sucesso!")
