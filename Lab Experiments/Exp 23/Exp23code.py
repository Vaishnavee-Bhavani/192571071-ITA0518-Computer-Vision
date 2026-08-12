# UNSHARP MASKING

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

# Blur Image
blurred = cv2.GaussianBlur(img_rgb,(9,9),0)

# Create Mask
mask = cv2.subtract(img_rgb, blurred)

# Sharpened Image
sharpened = cv2.add(img_rgb, mask)

# Display
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask)
plt.title("Unsharp Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(sharpened)
plt.title("Sharpened")
plt.axis("off")

plt.show()
