%cd /home/hasan/drone_p4_implementation/

from ultralytics import YOLO


model = YOLO("/home/hasan/drone_p4_implementation/runs/detect/train5/weights/best.pt")

results = model( "/home/hasan/drone_p4_implementation/hardware_videos/vd9.mp4", conf=0.3,save=True, show=True, device=2)


%cd /home/hasan/drone_p4_implementation/


from ultralytics import YOLO
import cv2
import os

# Paths and parameters
processed_video_path = "/home/hasan/drone_p4_implementation/videos/v5e_det.avi"
output_folder_processed = "/home/hasan/drone_p4_implementation/det_frames_3/"


original_video_path =  "/home/hasan/drone_p4_implementation/videos/v5e.mp4"
output_folder_original ="/home/hasan/drone_p4_implementation/org_frames3/"


os.makedirs(output_folder_processed, exist_ok=True)
os.makedirs(output_folder_original, exist_ok=True)

# Load YOLO model
model = YOLO("/home/hasan/drone_p4_implementation/runs/detect/train5/weights/best.pt")

# Parameters
confidence_threshold = 0.3  # Confidence threshold for detections

# Function to save all frames from a video (without bounding boxes for detection video)
def save_all_frames(video_path, output_folder, model=None, confidence_threshold=0.3, apply_detection=False):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run object detection only if requested
        if apply_detection:
            results = model(frame, conf=confidence_threshold)
            accident_or_fire_detected = False
            for result in results:
                for box in result.boxes:
                    class_label = model.names[int(box.cls)]
                    if class_label in ['0', '1']:  # Assuming class '0' is accident, '1' is car fire
                        accident_or_fire_detected = True
                        break
                if accident_or_fire_detected:
                    break

        # Save each frame to the output folder (no skipping frames anymore)
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame: {frame_filename}")

    cap.release()

# Function to extract frames from the original video based on timestamps from the detection video
def extract_frames_from_original(video_path, output_folder, detection_video_path, confidence_threshold=0.3):
    # Open both detection and original video
    detection_cap = cv2.VideoCapture(detection_video_path)
    original_cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while detection_cap.isOpened():
        ret_detection, frame_detection = detection_cap.read()
        ret_original, frame_original = original_cap.read()

        if not ret_detection or not ret_original:
            break

        frame_count += 1

        # Save frame from detection video
        frame_filename_detection = os.path.join(output_folder, f"detection_frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename_detection, frame_detection)
        print(f"Saved detection frame: {frame_filename_detection}")

        # Save frame from original video (synchronized with detection frame)
        frame_filename_original = os.path.join(output_folder, f"original_frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename_original, frame_original)
        print(f"Saved original frame: {frame_filename_original}")

    detection_cap.release()
    original_cap.release()

# Step 1: Save all frames from the detection video (with no skipping frames)
save_all_frames(processed_video_path, output_folder_processed, model=model, confidence_threshold=confidence_threshold, apply_detection=False)

# Step 2: Save all synchronized frames from both detection and original videos
extract_frames_from_original(original_video_path, output_folder_original, processed_video_path, confidence_threshold=confidence_threshold)

print("Frame extraction completed for both videos.")


