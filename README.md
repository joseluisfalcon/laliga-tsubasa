# LaLiga Tsubasa: Real-Time Football-to-Anime Transformation

![Captain Tsubasa](https://i.blogs.es/cd065e/captain-tsubasa/500_333.jpeg)

## Project Overview
LaLiga Tsubasa is an advanced AI-powered video processing pipeline designed to transform live football broadcasts into a stylized "Captain Tsubasa" anime aesthetic. The project leverages state-of-the-art neural networks to achieve consistent, high-quality cartoonization in near real-time.

### Core Objective
The primary goal is to provide an immersive, nostalgic experience by converting professional football footage into a retro video game/anime style, specifically optimized for a **360p @ 15 FPS** "Retro" look.

---

## Technical Stack & Algorithms

### 🎨 Stylization Engine: AnimeGANv3
We use **AnimeGANv3** (specifically the Hayao model) as our primary stylization backbone. Unlike traditional filters, AnimeGANv3 uses a Generative Adversarial Network (GAN) trained on specific artistic styles to transform real-world textures and colors into hand-drawn anime looks.

### ⚡ Optimization: NVIDIA TensorRT 10
To achieve high performance on modern GPUs (like the RTX 5070 Ti), we have integrated **NVIDIA TensorRT**. 
- **CUDA Cores**: The pipeline is optimized to run on NVIDIA Tensor Cores using FP16 precision.
- **Engine Caching**: The model is compiled into a native TensorRT engine, enabling instant startup and ultra-fast inference.
- **CPU Fallback**: The system is designed to be portable; if no compatible NVIDIA GPU is found, it automatically falls back to CUDA or standard CPU execution via ONNX Runtime.

### 🎮 Retro Aesthetic (360p @ 15 FPS)
Per user preference, the system targets a specific "Old Video Game" aesthetic:
- **Low Resolution (360p)**: Enhances the "pixelated" feel of classic anime games.
- **Low Framerate (15 FPS)**: Provides the classic "stuttery" animation style typical of traditional hand-drawn anime and 90s gaming consoles.
- **Pure Stylization**: All object detection layers (YOLO/Faces) have been removed in the final build to ensure a seamless, non-flickering global style application.

---

## Performance Metrics
On an RTX 5070 Ti, the optimized pipeline achieves:
- **81.18 FPS** (Global stylized inference at 360p).
- **0.18x Processing Ratio** (1 minute of video processed in approximately 11 seconds).

---

## External References
This project draws inspiration and logic from the following research and repositories:
1.  [U-2-Net](https://github.com/xuebinqin/U-2-Net): For saliency-based background/foreground patterns.
2.  [U-2-Net-StyleTransfer](https://github.com/Norod/U-2-Net-StyleTransfer): For integrating style transfer with structural segmentation.
3.  [ToonClip / ComicsHero](https://github.com/JacopoMangiavacchi/ToonClip-ComicsHero): For local face-to-toon transformation techniques.
4.  [AnimeGANv3 Official](https://github.com/TachibanaYoshino/AnimeGANv3): The core stylization weights and architecture.

---

## Installation & Usage
1. Ensure you have the `onnxruntime-gpu` and `tensorrt` libraries installed.
2. Prepare your source video in `video_samples/`.
3. Run the main processing script:
   ```bash
   python src/process_video.py
   ```
4. Find your stylized output in the `output/` directory.

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
*Created as part of the LaLiga Tsubasa Evolution Project - 2026*
