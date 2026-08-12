# LAPLACIAN SHARPENING (NEGATIVE CENTER)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload Image
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read Image
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Laplacian Mask
kernel = np.array([
    [0, 1, 0],
    [1,-4, 1],
    [0, 1, 0]
])

# Apply Mask
laplacian = cv2.filter2D(img_rgb, -1, kernel)

# Sharpened Image
sharpened = cv2.subtract(img_rgb, laplacian)

# Display
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sharpened)
plt.title("Sharpened Image")
plt.axis("off")

plt.show()
