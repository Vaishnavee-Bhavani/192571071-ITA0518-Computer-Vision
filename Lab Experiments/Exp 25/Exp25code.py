# HIGH BOOST MASKING USING FORMULA

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

# Blur
blurred = cv2.GaussianBlur(img_rgb,(9,9),0)

# Mask
mask = cv2.subtract(img_rgb, blurred)

# High Boost Factor
A = 2.5

# High Boost Result
highboost = cv2.addWeighted(img_rgb,1.0,mask,A,0)

# Display
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask)
plt.title("Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(highboost)
plt.title("High Boost Result")
plt.axis("off")

plt.show()
