# SHARPENING USING POSITIVE CENTER LAPLACIAN MASK

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# COLAB IMAGE UPLOAD
# =========================
from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Read Image
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Positive Center Laplacian Kernel
kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

# Sharpen
sharpened = cv2.filter2D(img_rgb, -1, kernel)

# Display
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sharpened)
plt.title("Positive Center Laplacian Sharpening")
plt.axis("off")

plt.show()
