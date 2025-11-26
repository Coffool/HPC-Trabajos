import time
import matplotlib.pyplot as plt
from PIL import Image
from sobel import SobelFilter

IMG_PATH = "utils/Charmander.png"

sobel = SobelFilter(mode="paralelo")
start_time = time.time()
result_img = sobel.apply_filter(IMG_PATH)
end_time = time.time()

original_img = Image.open(IMG_PATH)

print(f"Tiempo de ejecución {sobel.mode}: {end_time - start_time:.4f} segundos")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_img, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(result_img, cmap="gray")
axes[1].set_title("Sobel")
axes[1].axis("off")
plt.show()
