# =====================================================
# HARRIS CORNER DETECTION ANALYSIS USING OPENCV
# IMPROVED VERSION FOR REPORT OUTPUTS
# =====================================================

!pip install opencv-python matplotlib numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# =====================================================
# UPLOAD IMAGE
# =====================================================

print("Please upload an image for Harris Corner Detection")

uploaded = files.upload()

image_path = next(iter(uploaded))

# =====================================================
# READ IMAGE
# =====================================================

img = cv2.imread(image_path)

if img is None:
    print("Error loading image.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =====================================================
# CONVERT TO GRAYSCALE
# =====================================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_float = np.float32(gray)

# =====================================================
# HARRIS CORNER DETECTION
# =====================================================

harris = cv2.cornerHarris(
    gray_float,
    blockSize=2,
    ksize=3,
    k=0.04
)

# =====================================================
# CLEANER CORNER THRESHOLD
# =====================================================

corner_threshold = 0.08 * harris.max()

result = img.copy()

# Mark corners in RED
result[harris > corner_threshold] = [0, 0, 255]

result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# =====================================================
# COUNT CORNERS
# =====================================================

corner_count = np.count_nonzero(harris > corner_threshold)

print("\nTotal Corners Detected:", corner_count)

# =====================================================
# SAVE OUTPUT
# =====================================================

cv2.imwrite("harris_corners.jpg", result)

# =====================================================
# DISPLAY RESULTS
# =====================================================

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(result_rgb)
plt.title("Harris Corner Detection")
plt.axis("off")

plt.tight_layout()
plt.show()

# =====================================================
# BETTER CORNER RESPONSE MAP
# =====================================================

plt.figure(figsize=(8,6))

response_map = np.log(np.abs(harris) + 1)

plt.imshow(response_map, cmap='hot')
plt.title("Corner Response Map")
plt.colorbar()
plt.axis("off")

plt.show()

# =====================================================
# ANALYSIS
# =====================================================

print("\n========== ANALYSIS ==========")

print("Corner Count:", corner_count)

if corner_count < 500:
    print("Low number of corner features detected.")
elif corner_count < 3000:
    print("Moderate number of corner features detected.")
else:
    print("Large number of corner features detected.")

print("\nApplications:")
print("- Object Recognition")
print("- Image Matching")
print("- Panorama Stitching")
print("- Motion Tracking")
print("- Robotics and SLAM")

# =====================================================
# DOWNLOAD OUTPUT
# =====================================================

files.download("harris_corners.jpg")
