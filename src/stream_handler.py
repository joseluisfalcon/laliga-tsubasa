import cv2
import threading
import queue
import time

class StreamBuffer:
    def __init__(self, stream_url, buffer_seconds=30):
        self.stream_url = stream_url
        self.buffer_seconds = buffer_seconds
        self.frame_queue = queue.Queue()
        self.is_running = False
        self.fps = 0
        self.width = 0
        self.height = 0
        self.target_frames = 0
        
    def start(self):
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print(f"[*] Stream capture started from {self.stream_url}")

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            print(f"[ERROR] Could not open stream: {self.stream_url}")
            self.is_running = False
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0: self.fps = 30 # Fallback
        
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.target_frames = int(self.fps * self.buffer_seconds)
        
        print(f"[*] Stream metadata: {self.width}x{self.height} @ {self.fps} FPS")
        print(f"[*] Target buffer frames: {self.target_frames}")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Link lost. Retrying...")
                time.sleep(1)
                cap.open(self.stream_url)
                continue
            
            # Simple buffer management: if queue is full, drop oldest
            if self.frame_queue.qsize() >= self.target_frames:
                try: self.frame_queue.get_nowait()
                except queue.Empty: pass
            
            self.frame_queue.put(frame)
        
        cap.release()

    def get_batch(self, count=None):
        """Retrieve a specific number of frames from the buffer."""
        frames = []
        batch_size = count if count else self.frame_queue.qsize()
        
        for _ in range(batch_size):
            try:
                frames.append(self.frame_queue.get_nowait())
            except queue.Empty:
                break
        return frames

    def stop(self):
        self.is_running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()

if __name__ == "__main__":
    # Test with the local sample file as a "live stream" simulation
    sb = StreamBuffer("video_samples/barcelona_bilbao_01.mp4", buffer_seconds=5)
    sb.start()
    
    time.sleep(2) # Accumulate some frames
    print(f"[*] Frames in buffer: {sb.frame_queue.qsize()}")
    
    batch = sb.get_batch(10)
    print(f"[*] Retrieved {len(batch)} frames from buffer")
    
    sb.stop()
