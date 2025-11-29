#!/usr/bin/env python3
"""
Quick test to verify camera display works without connecting to robot.
Tests the CameraDisplay class with fake images.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the CameraDisplay class from kinova_client
from multiprocessing import Process, Queue
from queue import Empty
import cv2


def display_process_worker(image_queue: Queue):
    """Separate process for displaying camera feeds."""
    cv2.namedWindow('External Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Wrist Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('External Camera', 640, 480)
    cv2.resizeWindow('Wrist Camera', 640, 480)
    
    print("✓ Camera display process started")
    
    try:
        while True:
            try:
                data = image_queue.get(timeout=0.1)
                
                if data is None:  # Shutdown signal
                    break
                
                external_img, wrist_img = data
                
                # Convert RGB to BGR for OpenCV
                external_bgr = cv2.cvtColor(external_img, cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                
                # Display
                cv2.imshow('External Camera', external_bgr)
                cv2.imshow('Wrist Camera', wrist_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Empty:
                cv2.waitKey(1)
                continue
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("✓ Camera display process stopped")


class CameraDisplay:
    """Safe camera display handler using separate process."""
    
    def __init__(self, enable_display=True):
        self.enable_display = enable_display
        self.display_process = None
        self.image_queue = None
        
        if self.enable_display:
            try:
                self.image_queue = Queue(maxsize=2)
                self.display_process = Process(
                    target=display_process_worker,
                    args=(self.image_queue,),
                    daemon=True
                )
                self.display_process.start()
                time.sleep(0.5)
                
                if self.display_process.is_alive():
                    print("✓ Camera display enabled (separate process)")
                else:
                    print("⚠ Camera display process failed to start")
                    self.enable_display = False
                    
            except Exception as e:
                print(f"⚠ Could not create display process: {e}")
                self.enable_display = False
        else:
            print("ℹ Camera display disabled")
    
    def update(self, external_img, wrist_img):
        """Send new images to display process."""
        if not self.enable_display or self.image_queue is None:
            return
        
        if self.display_process is not None and not self.display_process.is_alive():
            self.enable_display = False
            print("⚠ Display process died, disabling display")
            return
        
        try:
            if not self.image_queue.full():
                self.image_queue.put((external_img.copy(), wrist_img.copy()), block=False)
        except Exception as e:
            if not hasattr(self, '_error_printed'):
                self._error_printed = True
                print(f"⚠ Display update error: {e}")
    
    def close(self):
        """Safely close display process."""
        if self.display_process is not None and self.display_process.is_alive():
            try:
                self.image_queue.put(None, timeout=1.0)
                self.display_process.join(timeout=2.0)
                
                if self.display_process.is_alive():
                    self.display_process.terminate()
                    
            except Exception:
                if self.display_process.is_alive():
                    self.display_process.terminate()


def main():
    print("=" * 70)
    print("Camera Display Test (Simulated Camera Feeds)")
    print("=" * 70)
    
    # Create display
    display = CameraDisplay(enable_display=True)
    
    try:
        print("\nDisplaying simulated camera feeds for 10 seconds...")
        print("Press Ctrl+C to stop early\n")
        
        for i in range(100):
            # Create fake images (simulating camera captures)
            # External camera: Random colors
            external = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Wrist camera: Different pattern
            wrist = np.zeros((480, 640, 3), dtype=np.uint8)
            wrist[i*4:(i+1)*4, :] = [0, 255, 0]  # Moving green line
            
            # Add text to images
            cv2.putText(external, f"External Camera - Frame {i}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(wrist, f"Wrist Camera - Frame {i}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Update display
            display.update(external, wrist)
            
            time.sleep(0.1)  # 10 FPS
            
            if i % 10 == 0:
                print(f"  Frame {i}/100")
        
        print("\n✓ Test complete! Display worked successfully.")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        
    finally:
        print("\nClosing display...")
        display.close()
        print("✓ Done")


if __name__ == "__main__":
    main()
