import onnxruntime as ort
import cv2
import numpy as np

class ToonClipHandler:
    def __init__(self, model_path='models/toonclip/model.onnx', device='cuda'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
    def process_face(self, face_img):
        # Preprocessing: resize to 1024x1024, normalize
        h, w = face_img.shape[:2]
        img = cv2.resize(face_img, (1024, 1024))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0) # Add batch dimension
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        output = outputs[0][0]
        
        # Postprocessing
        output = np.transpose(output, (1, 2, 0)) # CHW to HWC
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = cv2.resize(output, (w, h))
        return output
