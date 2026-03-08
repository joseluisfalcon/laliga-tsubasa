import cv2
import time
import os
from cartoonizer import Cartoonizer

def process_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: No se encuentra {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error al abrir el vídeo {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cartoonizer = Cartoonizer()
    
    target_w = 1280
    target_h = int(height * (target_w / width))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    print(f"Procesando {input_path}...")
    start_time = time.time()
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply the new advanced cartoonizer with masking
        processed_frame = cartoonizer.process_frame(frame, apply_mask=True)
        out.write(processed_frame)
        
        processed_count += 1
        if processed_count % 50 == 0:
            elapsed = time.time() - start_time
            curr_fps = processed_count / elapsed
            print(f"Frame {processed_count}/{total_frames} - FPS actual: {curr_fps:.2f}")

    end_time = time.time()
    total_elapsed = end_time - start_time
    final_fps = processed_count / total_elapsed
    
    cap.release()
    out.release()

    print("-" * 30)
    print(f"Rendimiento medio: {final_fps:.2f} FPS.")
    
    video_duration = total_frames / fps
    print(f"Ratio (Proceso/Duración): {total_elapsed / video_duration:.2f}")
    
    if total_elapsed < (video_duration / 2):
        print("¡OBJETIVO CUMPLIDO!")
    else:
        print("Objetivo no cumplido. Se requiere optimización.")

if __name__ == "__main__":
    input_file = "video_samples/segments/segment_60.mp4"
    output_file = "output/result_segment_60.mp4"
    os.makedirs("output", exist_ok=True)
    process_video(input_file, output_file)
