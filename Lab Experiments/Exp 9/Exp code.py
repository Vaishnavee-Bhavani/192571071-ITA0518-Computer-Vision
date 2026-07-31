import cv2
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
        # Built-in 90-degree rotations
        rotated_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rotated_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Use cv2_imshow for displaying images in Colab
        print("Original Image:")
        cv2_imshow(img)
        print("\nClockwise 90 Deg:")
        cv2_imshow(rotated_cw)
        print("\nCounter-Clockwise 90 Deg:")
        cv2_imshow(rotated_ccw)

        # cv2.waitKey(0) and cv2.destroyAllWindows() are not needed with cv2_imshow
else:
    print("No image selected or uploaded. Please upload an image to proceed.")
