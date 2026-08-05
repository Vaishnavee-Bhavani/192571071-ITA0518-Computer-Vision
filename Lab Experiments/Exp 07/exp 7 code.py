"""7. Capture video from web Camera and  Display the video, in slow motion and in fast motion."""
import os
import cv2
import urllib.request
from IPython.display import HTML, display

# Step 1: Download a standard sample video (simulating a webcam clip)
sample_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4"
input_video = "webcam_sample.mp4"

print("Downloading sample webcam video...")
urllib.request.urlretrieve(sample_url, input_video)
print("✓ Video downloaded successfully!\n")

# Step 2: Function to create speed variants (slow/fast motion)
def create_speed_variant(input_path, output_path, speed_factor):
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'.")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Adjust target FPS based on speed factor
    new_fps = fps * speed_factor
    
    temp_output = "temp_" + output_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, new_fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        
    cap.release()
    out.release()
    
    # Re-encode using ffmpeg to standard H.264 web codec for browser playback
    os.system(f'ffmpeg -y -i "{temp_output}" -vcodec libx264 -f mp4 "{output_path}" -loglevel quiet')
    
    if os.path.exists(temp_output):
        os.remove(temp_output)
    return True

# Step 3: Generate Slow-Motion and Fast-Motion videos
print("Processing speed variants...")
create_speed_variant(input_video, "slow_motion.mp4", speed_factor=0.5)  # 0.5x Slow
create_speed_variant(input_video, "fast_motion.mp4", speed_factor=2.0)  # 2.0x Fast

# Re-encode original video for reliable HTML display
os.system(f'ffmpeg -y -i "{input_video}" -vcodec libx264 -f mp4 "web_original.mp4" -loglevel quiet')

# Step 4: Display all three videos side-by-side in Google Colab
html_code = """
<div style="display: flex; gap: 15px; justify-content: center;">
    <div style="text-align: center;">
        <h4>Original Speed (1.0x)</h4>
        <video width="260" controls><source src="web_original.mp4" type="video/mp4"></video>
    </div>
    <div style="text-align: center;">
        <h4>Slow Motion (0.5x)</h4>
        <video width="260" controls><source src="slow_motion.mp4" type="video/mp4"></video>
    </div>
    <div style="text-align: center;">
        <h4>Fast Motion (2.0x)</h4>
        <video width="260" controls><source src="fast_motion.mp4" type="video/mp4"></video>
    </div>
</div>
"""
display(HTML(html_code))
