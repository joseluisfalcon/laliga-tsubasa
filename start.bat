@echo off
title LaLiga Tsubasa LIVE
echo 🚀 Iniciando LaLiga Tsubasa Live...
call .venv\Scripts\activate.bat
python src\stream_video.py
pause
