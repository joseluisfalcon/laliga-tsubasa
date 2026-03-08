import cv2
import time
import argparse
import os
import json
import re
import subprocess
import threading
from queue import Queue
from cartoonizer import Cartoonizer

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

class VideoCaptureThread:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.q = Queue(maxsize=10)
        self.stopped = False
        
    def start(self):
        t = threading.Thread(target=self._update, daemon=True)
        t.start()
        return self
        
    def _update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.q.put(frame)
            else:
                time.sleep(0.01)
                
    def read(self):
        # Return the LATEST frame and clear the queue
        frame = None
        while not self.q.empty():
            frame = self.q.get()
        return (True, frame) if frame is not None else (True, None)
        
    def release(self):
        self.stopped = True
        self.cap.release()

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
    
    # Handle separate URL and Key if present
    stream_url = rtmp_url
    if not stream_url and "live_url" in live_settings and "live_key" in live_settings:
        url = str(live_settings['live_url'])
        key = str(live_settings['live_key'])
        # Standard YouTube joining: just append the key
        # If there's a ?, it's usually part of the server URL. 
        # Most RTMP clients want: rtmp://server/app/key
        if url.endswith('/'):
            stream_url = f"{url}{key}"
        else:
            stream_url = f"{url}/{key}"
    elif not stream_url:
        stream_url = live_settings.get("rtmp_url", None)

    # Try to treat src as an integer if it looks like one
    try:
        src = int(src)
    except ValueError:
        pass

    print(f"Opening source: {src}")
    cap_thread = VideoCaptureThread(src).start()
    
    # Wait for first frame to get dimensions
    frame = None
    while frame is None:
        ret, frame = cap_thread.read()
        if not ret:
            print("Error: Could not read source")
            return
        time.sleep(0.1)

    h_orig, w_orig = frame.shape[:2]
    out_h = int(h_orig * (width / w_orig))

    cartoonizer = Cartoonizer()

    # Initialize FFmpeg process if rtmp_url is provided
    ffmpeg_proc = None
    if stream_url:
        print(f"--- BROADCASTING ENABLED ---")
        print(f"Target: {stream_url}")
        # Optimized for YouTube - Ultra Low Latency
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{width}x{out_h}",
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'h264_nvenc',
            '-pix_fmt', 'yuv420p',
            '-preset', 'p1',          # Fastest possible
            '-tune', 'ull',           # Ultra Low Latency
            '-rc', 'vbr',
            '-b:v', '2500k',
            '-maxrate', '3000k',
            '-bufsize', '5000k',
            '-g', str(fps * 2),       # 2 second GOP
            '-f', 'flv',
            stream_url
        ]
        # Direct stderr to stdout to catch errors in the background log
        log_file = open("stream_log.txt", "w")
        ffmpeg_proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log_file, stdout=log_file)

    window_name = f"LaLiga Tsubasa LIVE - Source: {src}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"--- STREAM STARTED ---")
    print(f"Source: {src}")
    print(f"Resolution: {width}x{out_h}")
    print(f"Target FPS: {fps}")
    print(f"Press 'q' to quit")
    
    count = 0
    prev_time = 0
    frame_interval = 1.0 / fps
    stream_start_time = time.time()
    
    try:
        while not cap_thread.stopped:
            ret, frame = cap_thread.read()
            if frame is None:
                # Wait a tiny bit for new frames if empty
                time.sleep(0.001)
                continue

            current_time = time.time()
            
            # Process frame as fast as possible
            processed = cartoonizer.process_frame(frame, target_width=width)
            
            # Show result locally
            cv2.imshow(window_name, processed)
            
            # Stream to FFmpeg
            if ffmpeg_proc:
                try:
                    ffmpeg_proc.stdin.write(processed.tobytes())
                except BrokenPipeError:
                    print("\nFFmpeg Pipe Broken. Check RTMP URL/Key.")
                    break
            
            # Calculate processing stats
            count += 1
            proc_time = time.time() - current_time
            if proc_time > 0:
                 avg_fps = count / (time.time() - stream_start_time + 0.0001)
                 print(f"\rProcessed: {count} | Latency: {proc_time*1000:.1f}ms | Avg FPS: {avg_fps:.1f}", end="")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStream stopped by user.")
    finally:
        cap_thread.release()
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
