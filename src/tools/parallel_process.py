import subprocess
import threading
import os
import time

def run_ffmpeg(command, label):
    print(f"[{label}] Starting...")
    start_time = time.time()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        print(f"[{label}] Finished successfully in {duration:.2f}s")
    except subprocess.CalledProcessError as e:
        print(f"[{label}] Error: {e.stderr}")

def main():
    input_file = "video_samples/barcelona_bilbao_01.mp4"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return

    # Job 1: 480p @ original FPS (30)
    cmd1 = [
        'ffmpeg', '-y',
        '-i', input_file,
        '-vf', 'scale=-2:480',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy',
        'video_samples/barcelona_bilbao_03_480p.mp4'
    ]

    # Job 2: 480p @ 15 FPS
    cmd2 = [
        'ffmpeg', '-y',
        '-i', input_file,
        '-vf', 'scale=-2:480,fps=fps=15',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy',
        'video_samples/barcelona_bilbao_04_480p_15fps.mp4'
    ]

    threads = []
    t1 = threading.Thread(target=run_ffmpeg, args=(cmd1, "480p-30fps"))
    t2 = threading.Thread(target=run_ffmpeg, args=(cmd2, "480p-15fps"))

    threads.append(t1)
    threads.append(t2)

    print("--- Starting Parallel Processing ---")
    for t in threads:
        t.start()

    for t in threads:
        t.join()
    print("--- All jobs completed ---")

if __name__ == "__main__":
    main()
