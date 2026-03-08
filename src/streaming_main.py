import cv2
import time
import os
import argparse
from stream_handler import StreamBuffer
from cartoonizer import Cartoonizer

def run_streaming_pipeline(input_source, output_folder, buffer_sec=30):
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Inicializar Buffer de Stream (Capture)
    sb = StreamBuffer(input_source, buffer_seconds=buffer_sec)
    sb.start()
    
    # 2. Inicializar Cartoonizer (GPU)
    cartoonizer = Cartoonizer()
    
    print(f"[*] Esperando a llenar el primer buffer de {buffer_sec}s...")
    
    # Wait for enough metadata or timeout
    start_wait = time.time()
    while sb.target_frames == 0 and time.time() - start_wait < 10:
        time.sleep(0.5)

    if sb.target_frames == 0:
        print("[ERROR] No se pudo obtener metadatos del stream.")
        sb.stop()
        return

    # Esperamos a que el buffer tenga al menos el 90% de los frames objetivo
    while sb.frame_queue.qsize() < (sb.target_frames * 0.9):
        print(f"[*] Buffer: {sb.frame_queue.qsize()}/{sb.target_frames} frames", end="\r")
        time.sleep(1)
    
    print(f"\n[*] Primer buffer lleno ({sb.frame_queue.qsize()} frames). Iniciando procesamiento...")
    
    interval_idx = 0
    try:
        while True:
            # 3. Obtener el lote de frames del buffer (30 segundos)
            frames = sb.get_batch(sb.target_frames)
            if not frames:
                print("[!] Buffer insuficiente, esperando a recargar...")
                time.sleep(5)
                continue
            
            print(f"[*] Procesando Intervalo {interval_idx} ({len(frames)} frames)...")
            
            # 4. Procesar el intervalo
            interval_output = f"{output_folder}/output_batch_{interval_idx}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Usar fps detectado por el buffer e imagen cartoonificada
            out = cv2.VideoWriter(interval_output, fourcc, sb.fps, (1280, 720)) 
            
            start_proc = time.time()
            frame_skip = 2 # Process 1 out of 2 frames to achieve target ~15 FPS if 30 is too heavy
            
            for i, frame in enumerate(frames):
                if i % frame_skip != 0:
                    continue
                proc_frame = cartoonizer.process_frame(frame, apply_mask=True)
                out.write(proc_frame)
            
            out.release()
            end_proc = time.time()
            
            proc_time = end_proc - start_proc
            # Adjust effective buffer duration for ratio calculation if skipping
            effective_fps = sb.fps / frame_skip
            print(f"[+] Intervalo {interval_idx} completado en {proc_time:.2f}s (FPS Eff: {len(frames)/frame_skip/proc_time:.2f})")
            
            interval_idx += 1
            if interval_idx >= 3: # Limite de prueba
                break
            
    except KeyboardInterrupt:
        print("[*] Deteniendo pipeline...")
    finally:
        sb.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="video_samples/barcelona_bilbao_01.mp4")
    parser.add_argument("--output", default="output/streaming")
    parser.add_argument("--buffer", type=int, default=30)
    args = parser.parse_args()
    
    run_streaming_pipeline(args.input, args.output, buffer_sec=args.buffer)
