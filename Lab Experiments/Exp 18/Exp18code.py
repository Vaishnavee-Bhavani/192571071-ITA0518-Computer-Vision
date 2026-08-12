# SOBEL EDGE DETECTION - Y AXIS

import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload Image
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read Image
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel Y
sobel_y = cv2.Sobel(gray,
                    cv2.CV_64F,
                    0,   # dx
                    1,   # dy
                    ksize=3)

sobel_y = cv2.convertScaleAbs(sobel_y)

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sobel_y, cmap='gray')
plt.title("Sobel Y")
plt.axis("off")

plt.show()
