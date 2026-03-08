@echo off
echo 🛑 Deteniendo procesos de emision...
pskill ffmpeg 2>nul || taskkill /F /IM ffmpeg.exe /T 2>nul
pskill python 2>nul || taskkill /F /IM python.exe /T 2>nul
echo ✓ Todo detenido.
pause
