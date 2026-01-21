import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ========================
# CONFIGURAÇÕES
# ========================
DATASET_PATH = "dataset/archive/UTKFace"
IMG_SIZE = 96
EPOCHS = 10
BATCH_SIZE = 32

# ========================
# CARREGAMENTO DO DATASET
# ========================
X = []
y = []

files = os.listdir(DATASET_PATH)

for i, file in enumerate(files):
    try:
        idade = int(file.split("_")[0])
        img_path = os.path.join(DATASET_PATH, file)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(idade)

        if i % 500 == 0:
            print(f"Processadas {i}/{len(files)} imagens")

    except:
        continue

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

print("Total de imagens:", len(X))

# ========================
# SPLIT
# ========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# MODELO CNN
# ========================
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="linear")  # REGRESSÃO
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ========================
# TREINAMENTO
# ========================
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ========================
# SALVAR MODELO
# ========================
os.makedirs("models", exist_ok=True)
model.save("models/idade_model.keras")

print("✅ Modelo treinado e salvo com sucesso!")
