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
            
        print(f"Initializing Cartoonizer on {self.device} (PURE STYLE + HYBRID RETRO FILTER)")
        
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

    def apply_retro_style(self, img, threshold=0.81, scatter=1, color_levels=17, edge_detection_laplacian_ksize=5, apply_morphological_smoothing=True):
        """
        Hybrid filter combining Avidemux quantization with Abner-style saturation and edge treatment.
        """
        # 1. Smoothing (Scatter / Bilateral-like)
        ksize = max(3, scatter * 2 + 1)
        img_smooth = cv2.medianBlur(img, ksize)
        
        # 2. Color Enhancement (Abner-inspired HSV boost)
        # Boost saturation to make it pop like a classic console game
        hsv = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.2 # Boost saturation by 20%
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 3. Color Quantization (Avidemux style)
        div = 256 // color_levels
        quantized = (img_boosted // div) * div + div // 2
        
        # 4. Advanced Edge Detection  
        gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)
        # Laplacian for thick, cartoonish lines 
        edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=edge_detection_laplacian_ksize)
        
        # Thresholding
        _, mask = cv2.threshold(edges, int(threshold * 255), 255, cv2.THRESH_BINARY_INV) 
        
        # Morphological smoothing for edges (removes noise, connects lines)
        if apply_morphological_smoothing:
            kernel = np.ones((2,2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 5. Final Integration
        result = cv2.bitwise_and(quantized, quantized, mask=mask)
        
        return result

    def process_frame(self, frame, target_width=None):
        h, w = frame.shape[:2]
        
        # Determine target dimensions
        if target_width is None:
            target_w = 1280
        else:
            target_w = target_width
            
        target_h = int(h * (target_w / w))
        img = cv2.resize(frame, (target_w, target_h))
        
        # 1. ANIME STYLE (AnimeGANv3)
        if self.animegan:
            cartoon = self.animegan.predict(img)
        else:
            # Fallback
            div = 64
            cartoon = (cv2.bilateralFilter(img, 9, 75, 75) // div) * div + div // 2

        # 2. HYBRID RETRO POST-PROCESSING
        # Parameters: threshold 0.81, scatter 1, levels 17
        #final_result = self.apply_retro_style(cartoon, threshold=0.81, scatter=1, color_levels=17)
        final_result = self.apply_retro_style(cartoon, threshold=0.90, scatter=1, color_levels=25, edge_detection_laplacian_ksize=5, apply_morphological_smoothing=True)

        return final_result

if __name__ == "__main__":
    cap = cv2.VideoCapture("video_samples/segments/segment_0.mp4")
    ret, frame = cap.read()
    if ret:
        c = Cartoonizer()
        result = c.process_frame(frame, target_width=640)
        cv2.imwrite("output/test_hybrid_retro.jpg", result)
        print("Test frame saved to output/test_hybrid_retro.jpg")
    cap.release()
