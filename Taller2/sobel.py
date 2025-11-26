from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
import time


class SobelFilter:
    """
    Clase que aplica el filtro de Sobel a una imagen en escala de grises.
    Permite ejecutar el procesamiento tanto de forma secuencial como en paralelo
    mediante el uso de multiprocessing.

    Atributos:
        mode (str): Modo de ejecución ('secuencial' o 'paralelo').
    """

    def __init__(self, mode="secuencial"):
        """
        Inicializa el filtro Sobel con el modo de ejecución deseado.

        Parámetros:
            mode (str): 'secuencial' o 'paralelo'. Define el método de ejecución.
        """
        assert mode in ["secuencial", "paralelo"], (
            "Modo no válido. Use 'secuencial' o 'paralelo'."
        )
        self.mode = mode
        self.Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def apply_filter(self, img_path):
        """
        Aplica el filtro Sobel a la imagen especificada según el modo elegido.

        Parámetros:
            img_path (str): Ruta del archivo de imagen a procesar.

        Retorna:
            PIL.Image.Image: Imagen procesada con el filtro de Sobel.
        """
        img = Image.open(img_path).convert("L")  # Convertir a escala de grises
        img_matrix = np.array(img)

        if self.mode == "secuencial":
            return self._secuential(img_matrix)
        else:
            return self._parallel(img_matrix)

    def _secuential(self, img_matrix):
        """
        Implementación secuencial del filtro Sobel.

        Recorre píxel por píxel la imagen sin contar los border, aplicando las
        máscaras de Sobel para calcular la magnitud del gradiente.

        Parámetros:
            img_matrix (np.ndarray): Matriz 2D con los valores de intensidad.

        Retorna:
            PIL.Image.Image: Imagen resultante después de aplicar el filtro.
        """
        rows, cols = img_matrix.shape
        grad = np.zeros((rows, cols))

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                gx = np.sum(self.Kx * img_matrix[i - 1 : i + 2, j - 1 : j + 2])
                gy = np.sum(self.Ky * img_matrix[i - 1 : i + 2, j - 1 : j + 2])
                grad[i, j] = min(255, np.sqrt(gx**2 + gy**2))

        grad = np.clip(grad, 0, 255).astype(np.uint8)
        return Image.fromarray(grad)

    def _process_chunk(self, args):
        """
        Procesa un bloque de la imagen.

        Esta función es llamada por cada proceso en paralelo. Aplica el filtro
        Sobel únicamente sobre las filas asignadas al proceso.

        Parámetros:
            args (tuple): Contiene la imagen, el rango de filas a procesar y
            los kernels Kx y Ky.

        Retorna:
            np.ndarray: Bloque de la imagen procesado.
        """
        img_matrix, start_row, end_row, Kx, Ky = args
        rows, cols = img_matrix.shape
        grad_chunk = np.zeros((end_row - start_row, cols))

        for i_local, i in enumerate(range(start_row, end_row)):
            if i == 0 or i == rows - 1:
                continue
            for j in range(1, cols - 1):
                gx = np.sum(Kx * img_matrix[i - 1 : i + 2, j - 1 : j + 2])
                gy = np.sum(Ky * img_matrix[i - 1 : i + 2, j - 1 : j + 2])
                grad_chunk[i_local, j] = min(255, np.sqrt(gx**2 + gy**2))

        return grad_chunk

    def _parallel(self, img_matrix):
        """
        Implementación paralela del filtro Sobel.

        Divide la imagen en bloques horizontales y asigna cada bloque
        a un proceso independiente. Se agregan filas extras para
        evitar errores en los bordes del filtro.

        Parámetros:
            img_matrix (np.ndarray): Matriz 2D con los valores de intensidad.

        Retorna:
            PIL.Image.Image: Imagen resultante después de aplicar el filtro.
        """
        rows, cols = img_matrix.shape
        n_cpus = cpu_count()
        chunk_size = rows // n_cpus
        chunks = []

        # División de la imagen en bloques con superposición de bordes
        for i in range(n_cpus):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != n_cpus - 1 else rows

            # Se amplían los límites de cada bloque para incluir los bordes necesarios
            if start > 0:
                start -= 1
            if end < rows:
                end += 1

            chunks.append((img_matrix, start, end, self.Kx, self.Ky))

        # Procesamiento paralelo de los bloques
        with Pool(n_cpus) as pool:
            results = pool.map(self._process_chunk, chunks)

        # Eliminación de filas solapadas y combinación final
        processed = []
        for i, chunk in enumerate(results):
            if i > 0:
                chunk = chunk[1:]  # Eliminar la primera fila solapada
            if i < len(results) - 1:
                chunk = chunk[:-1]  # Eliminar la última fila solapada
            processed.append(chunk)

        grad = np.vstack(processed)
        grad = np.clip(grad, 0, 255).astype(np.uint8)
        return Image.fromarray(grad)
