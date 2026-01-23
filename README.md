### aging detection

# 🧠 Análise de Envelhecimento Facial e Cuidados com a Pele

Aplicação web baseada em **Visão Computacional e Deep Learning** que analisa uma imagem facial enviada pelo usuário para:

- Estimar a idade aparente
- Identificar sinais de envelhecimento
- Analisar características da pele
- Gerar recomendações personalizadas de cuidados dermatológicos

Projeto desenvolvido no contexto da disciplina de **Processamento Digital de Imagens (PDI)**.

---

## 🎯 Objetivo

Demonstrar a aplicação prática de técnicas de:
- Processamento Digital de Imagens
- Redes Neurais Convolucionais (CNN)
- Análise visual automatizada
- Integração IA + Web

Tudo isso em um sistema funcional e acessível via navegador.

---

## 🧰 Tecnologias Utilizadas

### Inteligência Artificial
- Python 3
- TensorFlow
- Keras
- OpenCV
- NumPy

### Web
- Flask
- HTML5
- CSS3
- Jinja2

### Deploy
- Railway (recomendado)
- Render
- Hugging Face Spaces

---

## 📂 Estrutura do Projeto

projeto/
├── models/
│ └── idade_model3.keras
├── dataset/
│ ├── UTKFace/
│ ├── MORPH/
│ └── dataset_padronizado/
├── scripts/
│ ├── treino_modelo.py
│ └── padronizar_dataset.py
├── web/
│ ├── app.py
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│ └── static/
│ ├── uploads/
│ └── results/
├── requirements.txt
└── README.md


---

## 📊 Datasets Utilizados

### UTKFace
- Mais de 20 mil imagens faciais
- Idade incluída no nome do arquivo
- Grande diversidade étnica e etária

### MORPH II
- Dataset profissional
- Necessita padronização
- Idade extraída do nome do arquivo

---

## 🔄 Padronização dos Datasets

Para unificar os datasets foi criado um processo automático que:

- Detecta o rosto com Haar Cascade
- Converte para escala de cinza
- Redimensiona para 96x96
- Padroniza o nome do arquivo

Formato final:
    idade_id.jpg


---

## 🧠 Treinamento do Modelo

Modelo CNN para **regressão de idade**:

- Entrada: imagem facial (96x96, grayscale)
- Saída: idade estimada
- Função de perda: MSE
- Métrica: MAE

Resultados médios:
- MAE entre **4 e 6 anos**
- Dataset padronizado melhora significativamente a precisão

---

## 🧪 Análises de Pele Implementadas

Além da idade, o sistema realiza análises visuais simples:

### Rugas
- Laplaciano (detecção de bordas)

### Manchas
- Desvio padrão do canal L (espaço LAB)

### Oleosidade
- Média de brilho da pele

---

## 💄 Recomendações de Cuidados com a Pele

Com base nas análises, o sistema sugere cuidados como:
- Retinol
- Vitamina C
- Protetor solar
- Hidratantes específicos
- Produtos oil-free

As recomendações são totalmente automáticas.

---

## 🌐 Aplicação Web

### Funcionalidades
- Upload de imagem
- Detecção automática do rosto
- Marcação visual do rosto
- Exibição da idade estimada
- Recomendações dermatológicas
- Interface responsiva

### Rotas
- `/` → Página inicial
- `/upload` → Processamento da imagem

---

## 🚀 Deploy

### Railway (Recomendado)
- Deploy rápido
- Suporte nativo a Flask
- Melhor desempenho para TensorFlow
- Porta gerenciada automaticamente

### Render
- Funciona corretamente
- Pode apresentar cold start
- Exige atenção com paths de arquivos estáticos

### Hugging Face Spaces
- Ideal para demonstração
- Interface pronta para IA
- Menos controle do backend

---

## ⚠️ Observações Importantes

- GPU não é obrigatória
- TensorFlow roda em CPU
- Pastas `static/uploads` e `static/results` são criadas automaticamente
- Caminho correto para exibir imagens:
    ./static/results/arquivo.jpg

---

## 📌 Trabalhos Futuros

- Melhorar o modelo com EfficientNet
- Detecção de acne e rosácea
- Histórico de envelhecimento ao longo do tempo
- Versão mobile
- Integração com profissionais da área dermatológica

---

## 👨‍💻 Autor

Projeto desenvolvido por **Higor**  
Área: Processamento Digital de Imagens e Inteligência Artificial aplicada à saúde da pele
