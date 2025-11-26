# Taller 2: Detección de bordeado mediante algoritmo de Sobel
En este taller se implementa el filtro de sobel para detectar bordes en imágenes con ejecución en secuencial y paralelo.
## Estructura del proyecto

```bash
.
├── main.py #Archivo para la ejecución
├── README.md
├── sobel.py #Implementación secuencial y paralela de SOBEL
└── utils/ #Recursos complementarios
    ├── Charmander.png
    ├── Charmander_gray.png
    └── parallelGrayScale.py
```

## Ejecución
Primero se instalan los requerimientos con

```bash
pip install -r requirements.txt
```

Y luego de eso podemos ejecutar el programa con 

```bash
python3 main.py
```

Dentro de `main.py` podemos cambiar el modo de ejecución entre secuencial y paralelo de la siguiente forma

```python
sobel = SobelFilter(mode="secuencial")
sobel = SobelFilter(mode="paralelo")
```

Después de la ejecución el programa devolverá el tiempo de ejecución de este así como la comparación entre la imagen original y el bordeado obtenido de esta