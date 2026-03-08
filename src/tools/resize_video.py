import os
import subprocess
import sys

def resize_video(input_path, output_path, target_res="1280:720"):
    """
    Resizes a video using ffmpeg.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at {input_path}")
        return False
        
    print(f"Resizing {input_path} to {target_res}...")
    
    command = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', f'scale={target_res}',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'copy',
        output_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Success! Video saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion: {e}")
        return False

if __name__ == "__main__":
    # Default settings as per user request
    input_file = "video_samples/barcelona_bilbao_01.mp4"
    output_file = "video_samples/barcelona_bilbao_02_720p.mp4"
    
    # Allow overriding from CLI
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    resize_video(input_file, output_file)
