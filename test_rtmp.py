import cv2
import time
import subprocess
import os
import json
import re

def load_settings(path="settings.json"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        content = re.sub(r'^\s*//.*', '', content, flags=re.MULTILINE)
        content = re.sub(r',\s*([\]}])', r'\1', content)
        return json.loads(content, strict=False)

settings = load_settings()
live_settings = settings["live_stream"]
source = live_settings["source"]
width = 640
fps = 15
stream_url = f"{live_settings['live_url']}/{live_settings['live_key']}"

print(f"Connecting to: {source}")
cap = cv2.VideoCapture(source)
ret, frame = cap.read()
if not ret:
    print("Failed to read source")
    exit(1)

h, w = frame.shape[:2]
out_h = int(h * (width / w))

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
    '-preset', 'p1',
    '-f', 'flv',
    stream_url
]

print("Starting FFmpeg...")
ffmpeg_proc = subprocess.Popen(command, stdin=subprocess.PIPE)

print("Streaming raw frames for 10 seconds...")
start = time.time()
count = 0
try:
    while time.time() - start < 10:
        ret, frame = cap.read()
        if not ret: break
        resized = cv2.resize(frame, (width, out_h))
        ffmpeg_proc.stdin.write(resized.tobytes())
        count += 1
        if count % 15 == 0:
            print(f"Sent {count} frames...")
except Exception as e:
    print(f"Error: {e}")
finally:
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    cap.release()
print("Done.")
