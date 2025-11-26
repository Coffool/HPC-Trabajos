import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Process, shared_memory
import os

IMAGE = "Charmander.png"
WORKERS = 8


def worker(shm_name, shape, dtype, start, end):
    shm = shared_memory.SharedMemory(name=shm_name)
    result = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    for i in range(start, end):
        for j in range(shape[1]):
            pixel = result[i, j]
            gray_value = np.mean(pixel)
            result[i, j] = [gray_value, gray_value, gray_value]
    shm.close()


def image_to_grayscale_parallel(img, n_processes):
    inicio = time.perf_counter()
    height = len(img)
    split = height // n_processes

    shm = shared_memory.SharedMemory(create=True, size=img.nbytes)
    shared_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
    np.copyto(shared_img, img)

    processes = []
    for i in range(n_processes):
        start = i * split
        end = (i + 1) * split if i < n_processes - 1 else height
        p = Process(target=worker, args=(shm.name, img.shape, img.dtype, start, end))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    result_image = np.copy(shared_img)
    shm.close()
    shm.unlink()

    fin = time.perf_counter()
    execution_time = fin - inicio
    print(f"Tiempo de ejecución: {execution_time:.6f} segundos")

    return result_image, execution_time


if __name__ == "__main__":
    img = iio.imread(IMAGE)
    img_grayscale, execution_time = image_to_grayscale_parallel(img, WORKERS)

    # --- Mostrar ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(
        f"PARALLEL - Tiempo de ejecución: {execution_time:.4f} segundos, utilizando {WORKERS} procesos",
        fontsize=16,
    )
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[1].imshow(img_grayscale)
    axes[1].set_title("Resultado")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Guardar en el mismo directorio ---
    output_path = os.path.join(os.path.dirname(__file__), "Charmander_gray.png")
    iio.imwrite(output_path, img_grayscale.astype(np.uint8))
    print(f"Imagen guardada en: {output_path}")
