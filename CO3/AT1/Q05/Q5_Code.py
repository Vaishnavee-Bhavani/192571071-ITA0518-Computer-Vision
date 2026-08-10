# Effect of Noise on Harris Corner Detection

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# ===== Upload Image =====
uploaded = files.upload()

image_path = list(uploaded.keys())[0]

img = cv2.imread(image_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===== Function to Add Gaussian Noise =====

def add_gaussian_noise(image, sigma):

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)

# Create Noisy Images

low_noise = add_gaussian_noise(gray, 10)
medium_noise = add_gaussian_noise(gray, 25)
high_noise = add_gaussian_noise(gray, 50)

# ===== Harris Corner Function =====

def harris_corners(gray_image):

    gray_float = np.float32(gray_image)

    dst = cv2.cornerHarris(
        gray_float,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    dst = cv2.dilate(dst, None)

    result = cv2.cvtColor(
        gray_image,
        cv2.COLOR_GRAY2RGB
    )

    result[dst > 0.01 * dst.max()] = [255, 0, 0]

    corner_count = np.sum(
        dst > 0.01 * dst.max()
    )

    return result, corner_count

# Detect Corners

original_corners, c1 = harris_corners(gray)
low_corners, c2 = harris_corners(low_noise)
medium_corners, c3 = harris_corners(medium_noise)
high_corners, c4 = harris_corners(high_noise)

# Blur High Noisy Image and Detect Again

blurred = cv2.GaussianBlur(
    high_noise,
    (5, 5),
    0
)

improved_corners, c5 = harris_corners(blurred)

# Display Results

plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
plt.imshow(original_corners)
plt.title(f'Original\nCorners={c1}')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(low_corners)
plt.title(f'Low Noise\nCorners={c2}')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(medium_corners)
plt.title(f'Medium Noise\nCorners={c3}')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(high_corners)
plt.title(f'High Noise\nCorners={c4}')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(improved_corners)
plt.title(f'Blurred + Harris\nCorners={c5}')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print Counts

print("\nCorner Counts:")
print("Original     :", c1)
print("Low Noise    :", c2)
print("Medium Noise :", c3)
print("High Noise   :", c4)
print("After Blur   :", c5)
