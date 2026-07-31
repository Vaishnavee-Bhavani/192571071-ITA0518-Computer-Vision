import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow # Used for displaying static frames

# File upload
uploaded = files.upload()

video_path = None
if uploaded:
    # Assuming only one file is uploaded for simplicity
    video_name = list(uploaded.keys())[0]
    video_path = video_name
    print(f"Uploaded file: {video_name}")

if video_path:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}. Please check the file path.")
    else:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        output_filename = "output_transformed_video.mp4"
        # Using 'mp4v' for MP4 codec, ensure the output size matches the transformation output
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (300, 300))

        # Perspective Transformation Matrix (from original code)
        pts1 = np.float32([[100, 100], [400, 100], [100, 400], [400, 400]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        print(f"Processing video and saving transformed frames to {output_filename}...")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            transformed_frame = cv2.warpPerspective(frame, M, (300, 300))
            out.write(transformed_frame)

            # Optional: Display the first original and transformed frame for visual check
            if frame_count == 0:
                print("First frame (original):")
                cv2_imshow(frame)
                print("First frame (transformed):")
                cv2_imshow(transformed_frame)

            frame_count += 1

        cap.release()
        out.release()
        print(f"Video processing complete. Output saved to {output_filename}. You can download it from the 'files' tab.")

        # No need for cv2.waitKey and cv2.destroyAllWindows in Colab when using this approach
else:
    print("No video selected or uploaded.")
