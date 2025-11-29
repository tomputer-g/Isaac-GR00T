#!/usr/bin/env python3
"""
Test script to check if OpenCV display works.
Run this to diagnose display issues before running the full client.
"""

import cv2
import numpy as np
import time
from multiprocessing import Process, Queue

def test_simple_display():
    """Test simple OpenCV display in main process."""
    print("\n=== Testing Simple Display (Main Process) ===")
    try:
        # Create a test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
        cv2.imshow('Test Window', img)
        
        print("✓ Window created successfully")
        print("  If you can see a window with random pixels, display works!")
        print("  Press any key in the window to continue...")
        
        cv2.waitKey(3000)  # Wait 3 seconds
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        print("✓ Simple display test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Simple display test FAILED: {e}")
        return False


def display_worker(queue):
    """Worker process for multiprocess display test."""
    cv2.namedWindow('Test Window MP', cv2.WINDOW_NORMAL)
    
    for i in range(10):
        try:
            img = queue.get(timeout=1.0)
            cv2.imshow('Test Window MP', img)
            cv2.waitKey(1)
        except:
            break
    
    cv2.destroyAllWindows()


def test_multiprocess_display():
    """Test OpenCV display in separate process."""
    print("\n=== Testing Multiprocess Display ===")
    try:
        queue = Queue()
        process = Process(target=display_worker, args=(queue,))
        process.start()
        
        # Send test images
        for i in range(10):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            queue.put(img)
            time.sleep(0.1)
        
        process.join(timeout=3.0)
        
        if process.is_alive():
            process.terminate()
        
        print("✓ Multiprocess display test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Multiprocess display test FAILED: {e}")
        return False


def main():
    print("=" * 70)
    print("OpenCV Display Test")
    print("=" * 70)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Display environment: {cv2.getBuildInformation().split('GUI:')[1].split()[0] if 'GUI:' in cv2.getBuildInformation() else 'Unknown'}")
    
    # Run tests
    simple_ok = test_simple_display()
    multiprocess_ok = test_multiprocess_display()
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)
    print(f"Simple Display:      {'✓ PASS' if simple_ok else '✗ FAIL'}")
    print(f"Multiprocess Display: {'✓ PASS' if multiprocess_ok else '✗ FAIL'}")
    print("=" * 70)
    
    if simple_ok and multiprocess_ok:
        print("\n✓ All tests passed! Camera display should work.")
        print("  Run kinova_client.py normally")
    elif multiprocess_ok:
        print("\n⚠ Only multiprocess display works.")
        print("  This is the method used by kinova_client.py")
    else:
        print("\n✗ Display tests failed!")
        print("  Solutions:")
        print("  1. Use --no-display flag: python kinova_client.py --no-display")
        print("  2. Enable X11 forwarding: ssh -X user@host")
        print("  3. Set DISPLAY variable: export DISPLAY=:0")
        print("  4. Run on machine with GUI")


if __name__ == "__main__":
    main()
