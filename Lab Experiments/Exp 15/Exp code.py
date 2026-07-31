import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow # For displaying images in Colab

def compute_dlt(src_pts, dst_pts):
    """Computes the 3x3 Homography Matrix using DLT (Singular Value Decomposition)."""
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

    A = np.array(A)
    # Solve Ah = 0 using SVD
    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape((3, 3))

    # Normalize matrix
    return H / H[2, 2]

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

        src_pts = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
        dst_pts = np.float32([[40, 40], [cols - 80, 10], [10, rows - 60], [cols - 40, rows - 10]])

        # Calculate transformation matrix manually via DLT algorithm
        H_dlt = compute_dlt(src_pts, dst_pts)
        dlt_img = cv2.warpPerspective(img, H_dlt, (cols, rows))

        print("Original Image:")
        cv2_imshow(img)
        print("DLT Transformation:")
        cv2_imshow(dlt_img)

        # No need for waitKey and destroyAllWindows in Colab when using cv2_imshow
else:
    print("No image selected or uploaded.")
