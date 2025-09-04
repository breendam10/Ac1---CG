# 📌 Diferenças e Principais Características

## 1. **Síntese de Imagens (Computação Gráfica)**

* **Definição:** Criação de imagens a partir de modelos matemáticos e geométricos, sem depender de uma imagem real como entrada.
* **Aspectos principais:** modelagem 3D, renderização, shaders, engines gráficas.
* **Exemplo de aplicação:** Renderização 3D de objetos virtuais em jogos ou animações.

👉 **Aplicação prática:**
Usar **PyOpenGL** ou **Blender API** para gerar um cubo 3D rotacionando.

```python
import matplotlib.pyplot as plt
import numpy as np

# Exemplo simples de um cubo em 3D usando matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Vértices do cubo
r = [-1, 1]
for s, e in [
    ([r[0], r[0], r[0]], [r[0], r[0], r[1]]),
    ([r[0], r[0], r[1]], [r[0], r[1], r[1]]),
    ([r[0], r[1], r[1]], [r[1], r[1], r[1]]),
    ([r[1], r[1], r[1]], [r[0], r[1], r[1]])
]:
    ax.plot3D(*zip(s, e), color="b")

plt.title("Exemplo de Síntese de Imagem (Cubo 3D)")
plt.show()
```

---

## 2. **Processamento de Imagens**

* **Definição:** Técnicas aplicadas sobre uma imagem digital já existente para **melhorar**, **filtrar** ou **extrair informações**.
* **Aspectos principais:** filtros de ruído, detecção de bordas, transformadas, segmentação.
* **Exemplo de aplicação:** Remoção de ruído em imagens médicas.

👉 **Aplicação prática:**
Usar **OpenCV** para aplicar filtro de suavização em uma imagem.

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('exemplo.jpg')
blur = cv2.GaussianBlur(img, (7,7), 0)

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.title("Processada - Suavização")
plt.show()
```

---

## 3. **Visão Computacional (Artificial)**

* **Definição:** Área que busca **reconhecer, interpretar e entender** imagens ou vídeos, simulando a percepção humana.
* **Aspectos principais:** reconhecimento de padrões, aprendizado de máquina, deep learning, detecção e classificação.
* **Exemplo de aplicação:** Detecção de objetos em tempo real (carros, pessoas, etc.).

👉 **Aplicação prática:**
Usar o modelo **YOLOv5** para detectar objetos em uma imagem.

```python
import torch
from PIL import Image

# Carregar modelo YOLOv5 pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Rodar em uma imagem
img = 'cachorro.jpg'
results = model(img)

# Mostrar resultados
results.show()
```

---

## 4. **Visualização Computacional**

* **Definição:** Representação visual de dados complexos para facilitar a análise, interpretação e tomada de decisão.
* **Aspectos principais:** gráficos interativos, dashboards, visualização científica.
* **Exemplo de aplicação:** Visualização de simulação de dados meteorológicos.

👉 **Aplicação prática:**
Usar **Matplotlib** para visualização volumétrica de dados.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

plt.contourf(X, Y, Z, cmap="viridis")
plt.colorbar()
plt.title("Visualização Computacional - Função Seno Radial")
plt.show()
```

---

# 📊 Conclusão

* **Síntese de Imagens** → cria do zero (modelagem/renderização).
* **Processamento de Imagens** → transforma uma imagem já existente.
* **Visão Computacional** → interpreta o conteúdo da imagem.
* **Visualização Computacional** → exibe dados complexos de forma visual.

Cada área possui **objetivos diferentes**, mas todas se complementam dentro da Computação Visual.
