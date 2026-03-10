import cv2
import time
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from cartoonizer import Cartoonizer
import torch

print(f"CUDA Available: {ort.get_device() if 'ort' in globals() else 'Checking...'}")
# actually let's just check torch
print(f"Torch CUDA: {torch.cuda.is_available()}")

c = Cartoonizer()
# Create a dummy frame (640x360)
frame = (np.random.rand(360, 640, 3) * 255).astype(np.uint8)

print("Warming up...")
for _ in range(5):
    c.process_frame(frame, target_width=640)

print("Benchmarking 100 frames...")
start = time.time()
for _ in range(100):
    c.process_frame(frame, target_width=640)
elapsed = time.time() - start

print(f"--- RESULTS ---")
print(f"Total time for 100 frames: {elapsed:.2f}s")
print(f"FPS: {100/elapsed:.2f}")
