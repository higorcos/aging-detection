import cv2
import numpy as np
import matplotlib.pyplot as plt

from preProcessamento import detectar_rosto
from extracaoPDI import indice_rugas, indice_textura
from modeloCNN import criar_modelo

# Caminho da imagem
imagem_path = "imagens_teste/rosto.png"

# Pré-processamento
rosto_input = detectar_rosto(imagem_path)

if rosto_input is None:
    print("Rosto não detectado.")
    exit()

# Converter para imagem 2D para PDI
rosto_img = (rosto_input[0,:,:,0] * 255).astype("uint8")

# Análise PDI
rugas, bordas = indice_rugas(rosto_img)
textura, laplaciano = indice_textura(rosto_img)

print("Índice de Rugas:", rugas)
print("Índice de Textura:", textura)

# Modelo CNN
model = criar_modelo()
pred = model.predict(rosto_input)

classes = ["Jovem", "Envelhecimento Moderado", "Envelhecimento Avançado"]
resultado = classes[np.argmax(pred)]

print("Classificação CNN:", resultado)

# Visualização
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Rosto")
plt.imshow(rosto_img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Rugas")
plt.imshow(bordas, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Textura")
plt.imshow(laplaciano, cmap='gray')
plt.axis('off')

plt.show()