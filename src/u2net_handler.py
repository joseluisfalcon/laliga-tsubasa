import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
import os

# We need the U2Net model definition. Since we don't have the repo cloned, 
# we'll implement a simplified version or load via torchhub if available, 
# but for now, let's assume we need to implement the wrapper.

# This is a placeholder for the U2Net model logic which usually requires the model class.
# I will implement a minimal wrapper that assumes the model architecture is available or 
# I will try to load it if possible.

from u2net_lib import U2NET

class U2NetHandler:
    def __init__(self, model_path='models/u2net/u2net.pth', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = U2NET(3, 1)
        print(f"Loading U2Net weights from {model_path}...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        # Image is BGR, needs to be RGB and normalized
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = cv2.resize(img, (320, 320))
        img = img.astype(np.float32)
        
        # Normalization (standard for U2Net)
        tmp_img = np.zeros((img.shape[0], img.shape[1], 3))
        img = img / np.max(img)
        tmp_img[:,:,0] = (img[:,:,0]-0.485)/0.229
        tmp_img[:,:,1] = (img[:,:,1]-0.456)/0.224
        tmp_img[:,:,2] = (img[:,:,2]-0.406)/0.225
        
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = torch.from_numpy(tmp_img).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = self.model(tmp_img)
            
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        predict = pred.squeeze().cpu().numpy()
        
        predict = cv2.resize(predict * 255, (w, h)).astype(np.uint8)
        return predict
