import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def idade_para_classe(idade):
    if idade <= 12:
        return 0  # Criança
    elif idade <= 30:
        return 1  # Jovem
    elif idade <= 50:
        return 2  # Moderado
    else:
        return 3  # Avançado

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

X = []
y = []

dataset_path = "dataset/archive/UTKFace"


for file in os.listdir(dataset_path):
    try:
        idade = int(file.split("_")[0])
        classe = idade_para_classe(idade)

        img_path = os.path.join(dataset_path, file)
        img_color = cv2.imread(img_path)

        if img_color is None:
            continue

        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            continue

        x, y0, w, h = faces[0]
        rosto = gray[y0:y0+h, x:x+w]

        rosto = cv2.resize(rosto, (128, 128))
        rosto = rosto / 255.0

        X.append(rosto)
        y.append(classe)

    except:
        continue

X = np.array(X).reshape(-1, 128, 128, 1)
y = to_categorical(y, 4)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential([
    Input(shape=(128,128,1)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(X_train)

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_val, y_val)
)

os.makedirs("models", exist_ok=True)
model.save("models/cnn_envelhecimento_melhorado.h5")

print("Modelo melhorado treinado e salvo!")