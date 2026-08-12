# SOBEL EDGE DETECTION - X AND Y

import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload Image
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read Image
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel X
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Y
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Combine X and Y
sobel_xy = cv2.magnitude(sobel_x, sobel_y)

sobel_xy = cv2.convertScaleAbs(sobel_xy)

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sobel_xy, cmap='gray')
plt.title("Sobel XY")
plt.axis("off")

plt.show()
