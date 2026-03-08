#!/bin/bash
# LaLiga Tsubasa - Stop Stream
echo "🛑 Deteniendo procesos de emisión..."
pskill ffmpeg 2>/dev/null || taskkill //F //IM ffmpeg.exe //T 2>/dev/null
pskill python 2>/dev/null || taskkill //F //IM python.exe //T 2>/dev/null
echo "✓ Todo detenido."
