import cv2
import torch
import time
import os
from tqdm import tqdm
from cartoonizer import Cartoonizer

def process_video_30s(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate target dimensions (same as cartoonizer)
    target_w = 1280
    target_h = int(height * (target_w / width))

    # Define codec and output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    # Initialize Cartoonizer
    cartoonizer = Cartoonizer()

    print(f"Processing {total_frames} frames from {input_path}...")
    start_time = time.time()

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed = cartoonizer.process_frame(frame)
        
        # Write to output
        out.write(processed)

    end_time = time.time()
    cap.release()
    out.release()

    duration = end_time - start_time
    print(f"Processing finished in {duration:.2f}s ({total_frames / duration:.2f} fps)")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    if not os.path.exists('output'):
        os.makedirs('output')
    
    input_file = "video_samples/test_30s.mp4"
    output_file = "output/cartoon_30s_test.mp4"
    
    if os.path.exists(input_file):
        process_video_30s(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")
