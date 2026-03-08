import subprocess
import json
import os
import re

def load_settings(path='settings.json'):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        # Remove // comments ONLY at start of lines (handling potential CRLF)
        content = re.sub(r'^\s*//.*', '', content, flags=re.MULTILINE)
        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',\s*([\]}])', r'\1', content)
        return json.loads(content, strict=False)

try:
    settings = load_settings()
    live_settings = settings.get('live_stream', {})

    RTMP_URL = live_settings.get('live_url', "rtmp://a.rtmp.youtube.com/live2")
    STREAM_KEY = live_settings.get('live_key', "")
    DESTINATION = f"{RTMP_URL}/{STREAM_KEY}"

    # Use an existing test video
    INPUT_VIDEO = "video_samples/barcelona_bilbao_06_360p_30fps.mp4"

    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: {INPUT_VIDEO} not found.")
        exit(1)

    # Comando FFmpeg (User's parameters)
    command = [
        'ffmpeg',
        '-re',                       
        '-i', INPUT_VIDEO,           
        '-vcodec', 'libx264',        
        '-preset', 'veryfast',       
        '-maxrate', '3000k',         
        '-bufsize', '6000k',
        '-pix_fmt', 'yuv420p',       
        '-g', '50',                  
        '-acodec', 'aac',            
        '-f', 'flv',                 
        DESTINATION                  
    ]

    print(f"Iniciando directo de prueba en: {DESTINATION}")
    with open('test_stream_log.txt', 'w') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, stdout=log)
        print("Stream running... check YouTube Studio. Press Ctrl+C to stop.")
        process.wait()
except Exception as e:
    print(f"Error: {e}")
