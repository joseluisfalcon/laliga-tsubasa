# LaLiga Tsubasa: Real-Time Football-to-Anime Transformation

![Captain Tsubasa](https://i.blogs.es/cd065e/captain-tsubasa/500_333.jpeg)

## Project Overview
LaLiga Tsubasa is an advanced AI-powered video processing pipeline designed to transform live football broadcasts into a stylized "Captain Tsubasa" anime aesthetic. The project leverages state-of-the-art neural networks to achieve consistent, high-quality cartoonization in real-time.

### Core Objective
The primary goal is to provide an immersive, nostalgic experience by converting professional football footage into a retro video game/anime style, specifically optimized for a **360p @ 15 FPS** "Retro" look.

---

## Technical Stack & Algorithms

### 🎨 Stylization Engine: AnimeGANv3
We use **AnimeGANv3** (specifically the Hayao model) as our primary stylization backbone. Unlike traditional filters, AnimeGANv3 uses a Generative Adversarial Network (GAN) trained on specific artistic styles to transform real-world textures and colors into hand-drawn anime looks.

### ⚡ Optimization: NVIDIA TensorRT
To achieve high performance on modern GPUs (like the RTX 5070 Ti), we have integrated **NVIDIA TensorRT**. 
- **CUDA Cores**: The pipeline is optimized to run on NVIDIA Tensor Cores using FP16 precision.
- **Engine Caching**: The model is compiled into a native TensorRT engine, enabling instant startup and ultra-fast inference (~16ms per frame).
- **Ultra-Low Latency**: Configured for real-time broadcasting with specialized FFmpeg parameters (`-preset p1`, `-tune ull`).

### 🎮 Retro Aesthetic (360p @ 15 FPS)
Per user preference, the system targets a specific "Old Video Game" aesthetic:
- **Low Resolution (360p)**: Enhances the "pixelated" feel of classic anime games.
- **Low Framerate (15 FPS)**: Provides the classic "stuttery" animation style typical of traditional hand-drawn anime.

---

## Installation & Configuration

### ⚠️ IMPORTANT: Git LFS Requirement
This repository uses **Git Large File Storage (LFS)** for neural network weights and models. To clone the repository correctly, you **must** have Git LFS installed:
1. Install Git LFS: `git lfs install`
2. Clone the repository: `git clone https://github.com/joseluisfalcon/laliga-tsubasa.git`

### Prerequisites
- **NVIDIA GPU** (RTX 30 series or higher recommended).
- **NVIDIA Drivers**: Version 595.71+ (supports CUDA 13.2).
- **Python 3.10+** and a virtual environment.

### Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your credentials in `settings.json` (use `settings.json.SAMPLE` as a template).

---

## Usage

### 🚀 Quick Start (Scripts)
The project includes automated scripts for easy management:
- **Windows**: Double-click `start.bat` to launch or `stop.bat` to terminate.
- **Linux/Git Bash**: Run `./start.sh` or `./stop.sh`.

### 📡 Live Streaming
The system supports broadcasting to multiple platforms (YouTube, Twitch, etc.):
- **YouTube**: Set `live_url` to `rtmp://a.rtmp.youtube.com/live2`.
- **Twitch**: Set `live_url` to `rtmp://live.twitch.tv/app/`.

### 🧪 Development & Tests
- **Benchmarks**: Located in `src/benchmarks/` to measure FPS and latency.
- **Tests**: Connectivity and RTMP tests in `src/tests/`.

---

## Project Structure
- `src/`: Core Python logic and handlers.
- `models/`: Neural network weights (managed by Git LFS).
- `logs/`: Runtime logs and FFmpeg output.
- `video_samples/`: Input videos for testing.
- `output/`: Processed video files.

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
*Created as part of the LaLiga Tsubasa Evolution Project - 2026*
