import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow # For displaying images in Colab

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

        # At least 4 corresponding point pairs
        src_pts = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
        dst_pts = np.float32([[50, 50], [cols - 100, 20], [20, rows - 50], [cols - 50, rows - 20]])

        # Find Homography matrix H
        H, status = cv2.findHomography(src_pts, dst_pts)
        homography_img = cv2.warpPerspective(img, H, (cols, rows))

        print("Original Image:")
        cv2_imshow(img)
        print("Homography Transformation:")
        cv2_imshow(homography_img)

        # No need for waitKey and destroyAllWindows in Colab when using cv2_imshow
else:
    print("No image selected or uploaded.")
