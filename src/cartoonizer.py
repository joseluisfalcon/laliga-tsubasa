import torch
import cv2
import numpy as np
from ultralytics import YOLO
import warnings

# Ignore the sm_120 compatibility warning
warnings.filterwarnings("ignore", category=UserWarning)

from u2net_handler import U2NetHandler
from toonclip_handler import ToonClipHandler

class Cartoonizer:
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing Cartoonizer on {self.device}")
        
        # 1. Base Detectors
        self.detector = YOLO('yolo11n.pt').to(self.device)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 2. Advanced AI (U-2-Net & ToonClip)
        try:
            self.u2net = U2NetHandler(device=self.device)
            self.toonclip = ToonClipHandler(device=self.device)
            print("[INFO] Advanced AI models loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load advanced models: {e}")
            self.u2net = None
            self.toonclip = None

    def process_frame(self, frame, apply_mask=True):
        h, w = frame.shape[:2]
        target_w = 1280
        target_h = int(h * (target_w / w))
        img = cv2.resize(frame, (target_w, target_h))
        
        # A. SALIENCY SEGMENTATION (U-2-Net)
        # Identify main subjects (players, ball) to keep them vibrant
        if self.u2net:
            mask = self.u2net.predict(img)
            # Refine mask for silhoutte look
            _, mask_inv = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY_INV)
        else:
            mask = np.ones((target_h, target_w), dtype=np.uint8) * 255

        # B. CARTOON STYLE (Global)
        # 1980s Anime Look: Vibrant, High Saturation, Posterized
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] *= 1.5 # Saturate
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        vibrant = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Cel shading effect
        div = 64
        posterized = (cv2.bilateralFilter(vibrant, 9, 75, 75) // div) * div + div // 2
        
        # Edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cartoon = cv2.bitwise_and(posterized, posterized, mask=edges)

        # C. HYBRID FACE TRANSFORMATION (ToonClip)
        if apply_mask and self.toonclip:
            player_results = self.detector.predict(img, conf=0.3, verbose=False)
            for res in player_results:
                for box in res.boxes:
                    if int(box.cls[0]) != 0: continue # Player ONLY
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    head_h = int((y2 - y1) * 0.25)
                    head_roi_gray = gray[y1:y1+head_h, x1:x2]
                    
                    # Detect face inside player box
                    faces = self.face_cascade.detectMultiScale(head_roi_gray, 1.1, 3, minSize=(20, 20))
                    
                    for (fx, fy, fw, fh) in faces:
                        # Exact Face BBox
                        ax, ay = x1 + fx, y1 + fy
                        # Pad a bit for better Toon effect
                        pad = int(fw * 0.2)
                        fx1, fy1 = max(0, ax-pad), max(0, ay-pad)
                        fx2, fy2 = min(target_w, ax+fw+pad), min(target_h, ay+fh+pad)
                        
                        face_roi = img[fy1:fy2, fx1:fx2]
                        if face_roi.size > 0:
                            # Transform face using ToonClip
                            try:
                                toon_face = self.toonclip.process_face(face_roi)
                                cartoon[fy1:fy2, fx1:fx2] = toon_face
                            except:
                                pass # Fallback to global cartoon

        return cartoon
        
        return cartoon
        
        return cartoon

if __name__ == "__main__":
    cap = cv2.VideoCapture("video_samples/segments/segment_0.mp4")
    ret, frame = cap.read()
    if ret:
        c = Cartoonizer()
        result = c.process_frame(frame)
        cv2.imwrite("output/test_cartoon.jpg", result)
        print("Test frame saved to output/test_cartoon.jpg")
    cap.release()
