#funciona mais com erro

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def idade_para_classe(idade):
    if idade <= 30:
        return 0  # Jovem
    elif idade <= 50:
        return 1  # Moderado
    else:
        return 2  # Avançado

X = []
y = []

dataset_path = "dataset/archive/UTKFace"

for file in os.listdir(dataset_path):
    try:
        idade = int(file.split("_")[0])
        classe = idade_para_classe(idade)

        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (128, 128))
        img = img / 255.0

        X.append(img)
        y.append(classe)

    except:
        continue

X = np.array(X).reshape(-1, 128, 128, 1)
y = to_categorical(y, 3)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

os.makedirs("models", exist_ok=True)
model.save("models/cnn_envelhecimento_utkface.h5")

print("Modelo treinado e salvo com sucesso!")
