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
        rows, cols = img.shape[:2]

        # Select 3 non-collinear points from source image and map to destination points
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

        # 2x3 Affine transformation matrix
        M = cv2.getAffineTransform(pts1, pts2)
        affine_img = cv2.warpAffine(img, M, (cols, rows))

        cv2_imshow(img)
        cv2_imshow(affine_img)

        # No need for waitKey and destroyAllWindows in Colab when using cv2_imshow
else:
    print("No image selected or uploaded.")
