# HIGH BOOST SHARPENING

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

# High Boost Factor
A = 2

# High Boost Kernel
kernel = np.array([
    [0,-1,0],
    [-1,A+4,-1],
    [0,-1,0]
])

# Sharpen
highboost = cv2.filter2D(img_rgb,-1,kernel)

# Display
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(highboost)
plt.title(f"High Boost Sharpening (A={A})")
plt.axis("off")

plt.show()
