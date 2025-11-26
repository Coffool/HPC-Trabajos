# Taller 3: Procesamiento de video en algoritmos paralelos, transformando un video a escala de grises

Este proyecto implementa la conversión de videos a escala de grises utilizando Python. El cuaderno principal extrae frames de un video, los procesa para convertirlos a escala de grises y genera los resultados en carpetas separadas.

## Archivo Principal

`Video_Color_to_Gray_helper.ipynb`: Cuaderno principal que contiene todo el flujo de trabajo para:
- Cargar un video de entrada
- Extraer frames individuales
- Convertir cada frame a escala de grises
- Guardar los frames procesados

## Estructura del Proyecto

```
Taller3/
├── Video_Color_to_Gray_helper.ipynb  # Archivo principal del proyecto
├── utils/
│   ├── secuentialGrayScale.py        # Funciones auxiliares de conversión
│   └── cat_video.mp4                 # Video de ejemplo para pruebas
├── frames_video_original/            # Generado: frames extraídos del video original
├── frames_video_result/              # Generado: frames convertidos a escala de grises
└── video/                            # Carpeta para videos de entrada
```

### Nota sobre carpetas generadas

Las carpetas `frames_video_original/` y `frames_video_result/` son creadas automáticamente durante la ejecución del cuaderno y contienen los frames extraídos y procesados respectivamente.

## Uso

### Ejecución del Cuaderno Principal

1. Abre el archivo `Video_Color_to_Gray_helper.ipynb`

2. Ejecuta las celdas del cuaderno en orden:
   - Carga del video
   - Extracción de frames
   - Conversión a escala de grises
   - Visualización de resultados

3. Los frames originales se guardarán en `frames_video_original/`

4. Los frames procesados (escala de grises) se guardarán en `frames_video_result/`

## Detalles de Implementación

### Conversión a Escala de Grises

La conversión se realiza calculando el promedio de los tres canales RGB de cada píxel:

```python
gray_value = (R + G + B) / 3
```

Luego se replica este valor en los tres canales para mantener el formato RGB: `[gray, gray, gray]`

### Medición de Rendimiento

El proyecto utiliza `time.perf_counter()` para medir con precisión el tiempo de ejecución de las operaciones de conversión, lo cual es útil para comparar con futuras implementaciones optimizadas.

## Video de Ejemplo

El archivo `utils/cat_video.mp4` es un video de ejemplo incluido únicamente para propósitos de prueba. Puedes reemplazarlo con tus propios videos colocándolos en la carpeta `video/` o especificando la ruta directamente en el cuaderno.
