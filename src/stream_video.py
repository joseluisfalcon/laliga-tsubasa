import cv2
import time
import argparse
import os
import json
import re
import subprocess
from cartoonizer import Cartoonizer

def load_settings(path="settings.json"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        # Remove // comments
        content = re.sub(r'//.*', '', content)
        return json.loads(content)

def start_stream(source=None, target_width=None, target_fps=None, rtmp_url=None):
    """
    Starts a real-time stream. Prioritizes settings.json if source/width/fps are None.
    """
    settings = load_settings()
    live_settings = settings.get("live_stream", {}) if settings else {}

    # Prioritize arguments, then settings, then defaults
    src = source if source is not None else live_settings.get("source", 0)
    width = target_width if target_width is not None else live_settings.get("target_width", 640)
    fps = target_fps if target_fps is not None else live_settings.get("target_fps", 15)
    stream_url = rtmp_url if rtmp_url is not None else live_settings.get("rtmp_url", None)

    # Try to treat src as an integer if it looks like one
    try:
        src = int(src)
    except ValueError:
        pass

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Could not open source {src}")
        return

    # Optimization for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cartoonizer = Cartoonizer()
    
    # Get dimensions to calculate height
    ret, initial_frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame")
        return
    h_orig, w_orig = initial_frame.shape[:2]
    out_h = int(h_orig * (width / w_orig))

    # Initialize FFmpeg process if rtmp_url is provided
    ffmpeg_proc = None
    if stream_url:
        print(f"--- BROADCASTING ENABLED ---")
        print(f"Target: {stream_url}")
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{width}x{out_h}",
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-f', 'flv',
            stream_url
        ]
        ffmpeg_proc = subprocess.Popen(command, stdin=subprocess.PIPE)

    window_name = f"LaLiga Tsubasa LIVE - Source: {src}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"--- STREAM STARTED ---")
    print(f"Source: {src}")
    print(f"Resolution: {width}x{out_h}")
    print(f"Target FPS: {fps}")
    print(f"Press 'q' to quit")
    
    prev_time = 0
    frame_interval = 1.0 / fps
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or error reading frame.")
                break

            current_time = time.time()
            elapsed = current_time - prev_time
            
            # Simple frame rate control for display/streaming
            if elapsed >= frame_interval:
                prev_time = current_time
                
                # Process frame
                processed = cartoonizer.process_frame(frame, target_width=width)
                
                # Show result locally
                cv2.imshow(window_name, processed)
                
                # Stream to FFmpeg
                if ffmpeg_proc:
                    ffmpeg_proc.stdin.write(processed.tobytes())
                
                # Calculate processing stats
                proc_time = time.time() - current_time
                if proc_time > 0:
                     print(f"\rLatency: {proc_time*1000:.1f}ms | Render FPS: {1.0/(time.time()-current_time+0.0001):.1f}", end="")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStream stopped by user.")
    finally:
        cap.release()
        if ffmpeg_proc:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        cv2.destroyAllWindows()
        print("\n--- STREAM CLOSED ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaLiga Tsubasa Real-Time Streamer")
    parser.add_argument("--source", help="Camera index or Stream URL")
    parser.add_argument("--width", type=int, help="Target width")
    parser.add_argument("--fps", type=int, help="Target FPS")
    parser.add_argument("--rtmp", help="RTMP URL (YouTube/Twitch/etc.)")
    
    args = parser.parse_args()
    
    start_stream(source=args.source, target_width=args.width, target_fps=args.fps, rtmp_url=args.rtmp)
