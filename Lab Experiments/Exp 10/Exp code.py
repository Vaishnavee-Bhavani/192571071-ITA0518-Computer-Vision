import cv2
import numpy as np
# import tkinter as tk
# from tkinter import filedialog
from google.colab import files
from google.colab.patches import cv2_imshow # For displaying images in Colab

# File upload mechanism for Colab
print("Please upload an image file:")
uploaded = files.upload()

image_path = None
if uploaded:
    # Get the name of the first uploaded file
    image_name = next(iter(uploaded))
    image_path = image_name
    print(f"File '{image_name}' uploaded successfully.")
else:
    print("No file uploaded.")

if image_path:
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image from {image_path}. Please check the file path and format.")
    else:
        rows, cols = img.shape[:2]

        # Shift 100 pixels right (tx = 100) and 50 pixels down (ty = 50)
        tx, ty = 100, 50
        M = np.float32([[1, 0, tx], [0, 1, ty]])

        translated_img = cv2.warpAffine(img, M, (cols, rows))

        # Use cv2_imshow for displaying images in Colab
        print("Original Image:")
        cv2_imshow(img)
        print("\nMoved / Translated Image:")
        cv2_imshow(translated_img)

        # cv2.waitKey(0) and cv2.destroyAllWindows() are not needed with cv2_imshow
else:
    print("No image selected or uploaded. Please upload an image to proceed.")
