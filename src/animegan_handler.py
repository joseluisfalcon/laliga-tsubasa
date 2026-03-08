import cv2
import numpy as np
import onnxruntime as ort
import os

class AnimeGANHandler:
    def __init__(self, model_path=None, device='cuda'):
        # Try different possible names from the user's screenshot
        possible_paths = [
            'models/animeganv3/AnimeGANv3_Hayao_36.onnx',
            'models/animeganv3/AnimeGANv3_Shinkai_37.onnx',
            'models/animeganv3/animeganv3_hayao.onnx'
        ]
        
        selected_path = None
        if model_path and os.path.exists(model_path):
            selected_path = model_path
        else:
            for p in possible_paths:
                if os.path.exists(p):
                    selected_path = p
                    break
        
        if not selected_path:
            raise FileNotFoundError(f"AnimeGAN model not found in models/animeganv3/. Please download AnimeGANv3_Hayao_36.onnx or AnimeGANv3_Shinkai_37.onnx")
            
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        # Load ONNX model
        self.session = ort.InferenceSession(selected_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape
        self.input_shape = self.session.get_inputs()[0].shape # [1, H, W, 3] or [1, 3, H, W]
        print(f"AnimeGANv3 loaded on {device}. Input: {self.input_shape}")

    def predict(self, frame):
        # AnimeGANv3 usually expects RGB normalized to [-1, 1]
        input_h, input_w = 360, 640 # Typical for small ONNX models, otherwise dynamic
        
        # Resize if necessary
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Normalize to [-1, 1]
        img = (img / 127.5) - 1.0
        
        # Add batch dim [1, H, W, 3]
        img = np.expand_dims(img, axis=0)
        
        # Inference
        result = self.session.run([self.output_name], {self.input_name: img})[0]
        
        # Post-process: [-1, 1] -> [0, 255]
        out = (result[0] + 1.0) * 127.5
        out = np.clip(out, 0, 255).astype(np.uint8)
        
        # Return to BGR and original size
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (w, h))
