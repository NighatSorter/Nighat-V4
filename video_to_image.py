import cv2
import os

def extract_frames(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if not os.path.exists(video_name):
        os.makedirs(video_name)

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error opening video file")
        return

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            frame_filename = os.path.join(video_name, f"frame_{frame_count:04d}.jpg")    cv2.imwrite(frame_filename, frame)
    
        frame_count += 1

    video_capture.release()
    print(f"All frames saved in folder '{video_name}'")

video_path = 'mftl.mp4'
extract_frames(video_path)
