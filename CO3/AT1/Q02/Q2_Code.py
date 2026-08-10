# ==========================================
# IMAGE REPRESENTATION STUDY USING OPENCV
# ==========================================

# Install libraries (run once in Colab)
!pip install opencv-python matplotlib numpy

# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# ------------------------------------------
# STEP 1: UPLOAD IMAGE
# ------------------------------------------

print("Please upload an image file (jpg, png, jpeg)")
uploaded = files.upload()

# Get uploaded filename
image_path = next(iter(uploaded))

# ------------------------------------------
# STEP 2: READ IMAGE
# ------------------------------------------

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Convert BGR to RGB for displaying
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ------------------------------------------
# STEP 3: CONVERT TO GRAYSCALE
# ------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ------------------------------------------
# STEP 4: CONVERT TO BINARY
# ------------------------------------------

threshold_value = 127

_, binary = cv2.threshold(
    gray,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

# ------------------------------------------
# STEP 5: SAVE OUTPUTS
# ------------------------------------------

cv2.imwrite("grayscale_image.jpg", gray)
cv2.imwrite("binary_image.jpg", binary)

print("Grayscale image saved as grayscale_image.jpg")
print("Binary image saved as binary_image.jpg")

# ------------------------------------------
# STEP 6: DISPLAY RESULTS
# ------------------------------------------

plt.figure(figsize=(15,5))

# Original Image
plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

# Grayscale Image
plt.subplot(1,3,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

# Binary Image
plt.subplot(1,3,3)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# ------------------------------------------
# STEP 7: DOWNLOAD OUTPUT FILES
# ------------------------------------------

print("\nDo you want to download the generated images?")

files.download("grayscale_image.jpg")
files.download("binary_image.jpg")
