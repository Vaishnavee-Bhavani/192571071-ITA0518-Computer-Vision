# ==========================================
# EDGE DETECTION COMPARISON USING OPENCV
# Sobel vs Canny
# ==========================================

# Install required libraries (Run once)
!pip install opencv-python matplotlib numpy

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# ==========================================
# UPLOAD IMAGE
# ==========================================

print("Please upload an image...")
uploaded = files.upload()

image_path = next(iter(uploaded))

# ==========================================
# READ IMAGE
# ==========================================

img = cv2.imread(image_path)

if img is None:
    print("Error: Could not read image.")
    exit()

# Convert to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==========================================
# CONVERT TO GRAYSCALE
# ==========================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==========================================
# SOBEL EDGE DETECTION
# ==========================================

sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel_combined = cv2.magnitude(
    sobel_x,
    sobel_y
)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.convertScaleAbs(sobel_combined)

# ==========================================
# CANNY EDGE DETECTION
# ==========================================

canny = cv2.Canny(
    gray,
    100,
    200
)

# ==========================================
# SAVE OUTPUTS
# ==========================================

cv2.imwrite("grayscale.jpg", gray)
cv2.imwrite("sobel_x.jpg", sobel_x)
cv2.imwrite("sobel_y.jpg", sobel_y)
cv2.imwrite("sobel_combined.jpg", sobel_combined)
cv2.imwrite("canny_edges.jpg", canny)

print("Output images saved successfully!")

# ==========================================
# DISPLAY RESULTS
# ==========================================

plt.figure(figsize=(16,10))

plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(sobel_combined, cmap="gray")
plt.title("Sobel Combined")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(canny, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()

# ==========================================
# IMAGE STATISTICS
# ==========================================

sobel_edge_pixels = np.count_nonzero(sobel_combined)
canny_edge_pixels = np.count_nonzero(canny)

print("\n===== EDGE DETECTION ANALYSIS =====")
print("Sobel Edge Pixels :", sobel_edge_pixels)
print("Canny Edge Pixels :", canny_edge_pixels)

if canny_edge_pixels < sobel_edge_pixels:
    print("Observation: Canny generally produces thinner and cleaner edges.")
else:
    print("Observation: Sobel detected fewer edge pixels for this image.")

# ==========================================
# DOWNLOAD OUTPUT FILES
# ==========================================

print("\nDownloading generated output images...")

files.download("grayscale.jpg")
files.download("sobel_x.jpg")
files.download("sobel_y.jpg")
files.download("sobel_combined.jpg")
files.download("canny_edges.jpg")
