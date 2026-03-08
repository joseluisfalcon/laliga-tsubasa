import cv2
import time
import os
import json
import re
from cartoonizer import Cartoonizer

def process_video(input_path, output_path, target_width=None, target_fps=None):
    if not os.path.exists(input_path):
        print(f"Error: No se encuentra {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error al abrir el vídeo {input_path}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cartoonizer = Cartoonizer()
    
    # Calculate output dimensions
    out_w = target_width if target_width else width
    out_h = int(height * (out_w / width))
    
    # Calculate target FPS and frame skip
    if target_fps is None or target_fps >= source_fps:
        out_fps = source_fps
        frame_interval = 1
    else:
        out_fps = target_fps
        frame_interval = int(round(source_fps / target_fps))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))

    print(f"Procesando {input_path} -> {output_path} ({out_w}x{out_h} @ {out_fps}fps, interval: {frame_interval})...")
    start_time = time.time()
    processed_count = 0
    frames_read = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Corrected frame skipping logic
        if frames_read % frame_interval == 0:
            # Apply the pure animegan styler
            processed_frame = cartoonizer.process_frame(frame, target_width=out_w)
            out.write(processed_frame)
            processed_count += 1
            
            if processed_count % 50 == 0:
                elapsed = time.time() - start_time
                curr_fps = processed_count / elapsed
                print(f"Frame {processed_count} (input {frames_read}/{total_frames}) - FPS actual: {curr_fps:.2f}")

        frames_read += 1

    end_time = time.time()
    total_elapsed = end_time - start_time
    final_fps = processed_count / total_elapsed
    
    cap.release()
    out.release()

    print("-" * 30)
    print(f"Rendimiento medio: {final_fps:.2f} FPS.")
    
    video_duration = total_frames / source_fps
    print(f"Ratio (Proceso/Duración): {total_elapsed / video_duration:.2f}")
    
    if total_elapsed < video_duration:
        print("¡OBJETIVO CUMPLIDO (Faster than real-time)!")
    else:
        print("Procesado completado (Slower than real-time).")

def load_settings(path="settings.json"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        # Remove // comments ONLY at start of lines (handling potential CRLF)
        content = re.sub(r'^\s*//.*', '', content, flags=re.MULTILINE)
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',\s*([\]}])', r'\1', content)
        return json.loads(content, strict=False)

if __name__ == "__main__":
    settings = load_settings()
    
    if settings and "tests_codification" in settings:
        s = settings["tests_codification"]
        input_file = s["input"]["filename"]
        output_file = s["output"]["filename"]
        # Note: we can also extract animegan_model if needed
    else:
        # Fallback to defaults if settings.json is missing or invalid
        input_file = "video_samples/barcelona_bilbao_06_360p_30fps.mp4"
        output_file = "output/FINAL_TSUBASA_RETRO_360p_15fps_v10.mp4"

    os.makedirs("output", exist_ok=True)
    process_video(input_file, output_file, target_width=640, target_fps=15)
