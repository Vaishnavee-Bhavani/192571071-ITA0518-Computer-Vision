import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# File upload
uploaded = files.upload()

image_path = None
if uploaded:
    # Assuming only one file is uploaded for simplicity
    image_name = list(uploaded.keys())[0]
    image_path = image_name
    print(f"Uploaded file: {image_name}")

if image_path:
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}. Please check the file path and ensure it's a valid image.")
    else:
        # 4 corner points in input image -> 4 corner points in destination image
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

        # 3x3 Perspective transformation matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_img = cv2.warpPerspective(img, M, (300, 300))

        cv2_imshow(img)
        cv2_imshow(perspective_img)

        # No need for waitKey and destroyAllWindows in Colab when using cv2_imshow
else:
    print("No image selected or uploaded.")
