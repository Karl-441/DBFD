import cv2
import time
import argparse
import os
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    parser = argparse.ArgumentParser(description="DBFD Camera Utility")
    parser.add_argument("--index", type=int, default=config.CAMERA_INDEX, help="Camera index")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH, help="Frame Width")
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help="Frame Height")
    parser.add_argument("--record", action="store_true", help="Enable recording mode")
    args = parser.parse_args()

    print(f"Initializing Camera {args.index} ({args.width}x{args.height})...")
    
    if config.USE_LIBCAMERA:
        print("Initializing Libcamera (Picamera2)...")
        try:
            from core.camera_wrapper import LibCameraWrapper
            cap = LibCameraWrapper(args.width, args.height, config.FPS)
        except ImportError as e:
            print(f"Error: {e}")
            return
        
        # Libcamera wrapper doesn't have same props, assume success if no error
        print(f"Camera opened: {args.width}x{args.height} (Libcamera)")
        w, h = args.width, args.height
    else:
        cap = cv2.VideoCapture(args.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, config.FPS)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        # Check actual resolution
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera opened: {w}x{h} @ {fps} FPS")

    # Recording setup
    out = None
    if args.record:
        filename = os.path.join(config.OUTPUT_DIR, f"cam_test_{int(time.time())}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 10.0, (w, h))
        print(f"Recording to {filename}...")

    print("Controls:")
    print("  'q': Quit")
    print("  's': Save Snapshot")
    print("  'r': Toggle Recording")

    try:
        while True:
            # Polymorphic read
            if config.USE_LIBCAMERA:
                frame = cap.read()
                ret = frame is not None
            else:
                ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                time.sleep(1)
                continue

            # Display timestamp
            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, h-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("DBFD Camera Utility", frame)

            if out:
                out.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snap_path = os.path.join(config.OUTPUT_DIR, f"snapshot_{int(time.time())}.jpg")
                cv2.imwrite(snap_path, frame)
                print(f"Snapshot saved: {snap_path}")
            elif key == ord('r'):
                if out:
                    out.release()
                    out = None
                    print("Recording stopped.")
                else:
                    filename = os.path.join(config.OUTPUT_DIR, f"cam_test_{int(time.time())}.avi")
                    out = cv2.VideoWriter(filename, fourcc, 10.0, (w, h))
                    print(f"Recording started: {filename}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("Camera released.")

if __name__ == "__main__":
    main()
