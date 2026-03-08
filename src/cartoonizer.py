import torch
import cv2
import numpy as np
import warnings
import os

# Ignore the sm_120 compatibility warning
warnings.filterwarnings("ignore", category=UserWarning)

from animegan_handler import AnimeGANHandler

class Cartoonizer:
    def __init__(self, device=None, animegan_model=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing Cartoonizer on {self.device} (PURE STYLE MODE)")
        
        # Advanced AI
        try:
            # Use specific model if provided
            model_path = None
            if animegan_model:
                model_path = os.path.join('models', 'animeganv3', animegan_model)
                
            self.animegan = AnimeGANHandler(model_path=model_path, device=self.device) 
            print("[INFO] AnimeGANv3 model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load AnimeGAN model: {e}")
            self.animegan = None

    def process_frame(self, frame, target_width=None):
        h, w = frame.shape[:2]
        
        # Determine target dimensions
        if target_width is None:
            target_w = 1280
        else:
            target_w = target_width
            
        target_h = int(h * (target_w / w))
        img = cv2.resize(frame, (target_w, target_h))
        
        # ANIME STYLE (AnimeGANv3) - Pure implementation, no detection
        if self.animegan:
            cartoon = self.animegan.predict(img)
        else:
            # Fallback to simple cel shading
            div = 64
            cartoon = (cv2.bilateralFilter(img, 9, 75, 75) // div) * div + div // 2

        return cartoon

if __name__ == "__main__":
    cap = cv2.VideoCapture("video_samples/segments/segment_0.mp4")
    ret, frame = cap.read()
    if ret:
        c = Cartoonizer()
        result = c.process_frame(frame, target_width=640)
        cv2.imwrite("output/test_pure_style.jpg", result)
        print("Test frame saved to output/test_pure_style.jpg")
    cap.release()
