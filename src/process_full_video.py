import cv2
import torch
import time
import os
import sys
from tqdm import tqdm

# Add src to path to import cartoonizer
sys.path.append(os.path.join(os.getcwd(), 'src'))
from cartoonizer import Cartoonizer

def process_full_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate target dimensions
    target_w = 1280
    target_h = int(height * (target_w / width))

    # Define codec and output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    # Initialize Cartoonizer
    cartoonizer = Cartoonizer()

    print(f"Processing FULL video: {total_frames} frames from {input_path}...")
    start_global = time.time()

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed = cartoonizer.process_frame(frame)
        
        # Write to output
        out.write(processed)

    end_global = time.time()
    cap.release()
    out.release()

    total_duration = end_global - start_global
    fps_avg = total_frames / total_duration
    
    print("\n" + "="*40)
    print(f"RESULTADO DE PROCESAMIENTO")
    print("="*40)
    print(f"Tiempo total: {total_duration:.2f} segundos")
    print(f"Frames totales: {total_frames}")
    print(f"Rendimiento medio: {fps_avg:.2f} FPS")
    print(f"Salida: {output_path}")
    print("="*40)

if __name__ == "__main__":
    if not os.path.exists('output'):
        os.makedirs('output')
    
    input_file = "video_samples/barcelona_bilbao_01.mp4"
    output_file = "output/cartoon_full_video_final.mp4"
    
    if os.path.exists(input_file):
        process_full_video(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")
