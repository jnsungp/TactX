"""
Run tactile sensor publisher and recorder for different sensor types on each side.
Example: Daimon on right, FlexiTac on left.

Usage:
    python run_tactile_cross.py --left_type flexitac --right_type daimon --save_dir ~/tactile_data
"""

import threading
from dataclasses import dataclass
import tyro
import sys
import os
import time

# Add gello to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gello/tactile"))

from tactile_publisher import TactileSensorPublisher
from tactile_recorder import TactileRecorder

@dataclass
class Args:
    save_dir: str = "~/tactile_data"
    left_type: str = "flexitac"
    right_type: str = "daimon"

def main(args: Args):
    print(f"Starting Cross-Sensor Tactile System")
    print(f"  Left: {args.left_type}")
    print(f"  Right: {args.right_type}")
    print(f"  Save directory: {args.save_dir}")

    # Create publisher with cross configuration
    publisher = TactileSensorPublisher(
        sensor_type='cross',
        left_type=args.left_type,
        right_type=args.right_type
    )

    # Start publisher in background thread
    def run_publisher():
        try:
            publisher.start()
            publisher.publish_loop()
        except Exception as e:
            print(f"Publisher error: {e}")
        finally:
            publisher.stop()

    publisher_thread = threading.Thread(target=run_publisher, daemon=True)
    publisher_thread.start()

    # Wait for publisher to initialize
    timeout = 30.0
    start = time.time()
    while not publisher.ready and (time.time() - start) < timeout:
        if publisher.error:
            raise RuntimeError(f"Publisher failed to start: {publisher.error}")
        time.sleep(0.1)

    if not publisher.ready:
        raise RuntimeError("Publisher failed to start within timeout")

    print("Publisher ready, starting recorder...\n")

    # Create and run recorder (blocking, handles keyboard)
    recorder = TactileRecorder(
        sensor_type='cross',
        left_type=args.left_type,
        right_type=args.right_type,
        save_dir=args.save_dir,
    )

    try:
        recorder.run()
    finally:
        publisher.stop()
        print("Done.")

if __name__ == "__main__":
    main(tyro.cli(Args))
