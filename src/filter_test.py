import cv2
import numpy as np
import os

def apply_cartoon_sim(img, threshold=0.81, scatter=1, color_levels=17):
    # Parameters from user: threshold 0.81, scatter 1, color level 17
    
    # 1. Smoothing (Scatter simulates blurring/simplification)
    # Scatter 1 in these filters often corresponds to a small blur
    ksize = max(3, scatter * 2 + 1)
    img_smooth = cv2.medianBlur(img, ksize)
    
    # 2. Color Quantization
    # Reduces the number of colors to 'color_levels'
    div = 256 // color_levels
    quantized = (img_smooth // div) * div + div // 2
    
    # 3. Edge Detection
    # Gray scale for edge detection
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)
    # Laplacian or Adaptive Thresholding is commonly used in 'Cartoon' filters
    # We'll use a thresholded Laplacian to get the lines
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    
    # Apply the user's threshold
    # Higher threshold means fewer edges
    _, mask = cv2.threshold(edges, int((1.0 - threshold) * 255), 255, cv2.THRESH_BINARY_INV)
    
    # 4. Combine edges with quantized image
    # The mask has 0 for edges and 255 for areas
    result = cv2.bitwise_and(quantized, quantized, mask=mask)
    
    # Optional: Enhance colors slightly to match 'Hayao' style if needed
    # but the user likes the Avidemux specific look on top of the current output.
    
    return result

if __name__ == "__main__":
    # Test on a frame from the previously generated video
    input_video = "output/FINAL_TSUBASA_RETRO_360p_15fps.mp4"
    cap = cv2.VideoCapture(input_video)
    ret, frame = cap.read()
    if ret:
        # User parameters
        result = apply_cartoon_sim(frame, threshold=0.81, scatter=1, color_levels=17)
        
        output_path = "output/test_avidemux_replication.png"
        cv2.imwrite(output_path, result)
        print(f"Replication test saved to {output_path}")
    cap.release()
