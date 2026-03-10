import cv2
import time
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
source = settings["live_stream"]["source"]

print(f"Testing source: {source}")
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Failed to open source")
    exit(1)

start_time = time.time()
frames = 0
while frames < 60:
    ret, frame = cap.read()
    if not ret:
        print("End of stream")
        break
    frames += 1
    if frames % 10 == 0:
        elapsed = time.time() - start_time
        fps = frames / elapsed
        print(f"Received {frames} frames... Current FPS: {fps:.2f}")

total_elapsed = time.time() - start_time
print(f"--- RESULTS ---")
print(f"Total time: {total_elapsed:.2f}s")
print(f"Raw Source FPS: {frames/total_elapsed:.2f}")
cap.release()
