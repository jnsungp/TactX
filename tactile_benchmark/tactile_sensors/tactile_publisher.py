import time
import numpy as np
import sys
import os
import signal
# import wx
import threading
from multiprocessing import shared_memory

sys.path.insert(0, os.path.dirname(__file__))
from sensor_configs import SENSOR_MAP




class TactileSensorPublisher:
    def __init__(self, sensor_type='egain'):
        if sensor_type not in SENSOR_MAP:
            raise ValueError(f"Unknown sensor type: {sensor_type}. Available: {list(SENSOR_MAP.keys())}")

        self.sensor_type = sensor_type
        sensor_class, self.config = SENSOR_MAP[sensor_type]
        self.sensor = sensor_class()

        self.shm0 = None
        self.shm1 = None
        self.shm_array0 = None
        self.shm_array1 = None
        self.shm_created = False

        self.running = False
        self.cleanup_done = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}. Shutting down...")
        self.stop()
        sys.exit(0)

    def _init_shared_memory(self):
        print(f"Initializing shared memory for {self.sensor_type} sensors...")

        data_shape = self.config['data_shape']
        if data_shape is None:
            print("Sensor will manage its own shared memory")
            self.shm_created = False
            return

        element_count = int(np.prod(data_shape))
        element_size = np.dtype(np.float32).itemsize
        shm_size = int(element_count * element_size)

        try:
            self.shm0 = shared_memory.SharedMemory(
                name="tactile_s0_shm",
                create=True,
                size=shm_size
            )
            self.shm1 = shared_memory.SharedMemory(
                name="tactile_s1_shm",
                create=True,
                size=shm_size
            )
        except FileExistsError:
            print("Shared memory blocks already exist, connecting...")
            try:
                self.shm0 = shared_memory.SharedMemory(name="tactile_s0_shm")
                self.shm1 = shared_memory.SharedMemory(name="tactile_s1_shm")
            except Exception as e:
                print(f"Failed to connect to existing shared memory: {e}")
                print("Trying to unlink and recreate...")
                self._cleanup_shared_memory()
                self.shm0 = shared_memory.SharedMemory(
                    name="tactile_s0_shm",
                    create=True,
                    size=shm_size
                )
                self.shm1 = shared_memory.SharedMemory(
                    name="tactile_s1_shm",
                    create=True,
                    size=shm_size
                )

        self.shm_array0 = np.ndarray((element_count,), dtype=np.float32, buffer=self.shm0.buf)
        self.shm_array1 = np.ndarray((element_count,), dtype=np.float32, buffer=self.shm1.buf)

        self.shm_array0[:] = 0
        self.shm_array1[:] = 0

        self.shm_created = True
        print(f"Shared memory initialized: tactile_s0_shm and tactile_s1_shm ({shm_size} bytes each)")

    def _cleanup_shared_memory(self):
        if not self.shm_created:
            return

        try:
            if self.shm0:
                self.shm0.close()
                try:
                    self.shm0.unlink()
                except FileNotFoundError:
                    pass
                self.shm0 = None
        except Exception as e:
            print(f"Error cleaning up shm0: {e}")

        try:
            if self.shm1:
                self.shm1.close()
                try:
                    self.shm1.unlink()
                except FileNotFoundError:
                    pass
                self.shm1 = None
        except Exception as e:
            print(f"Error cleaning up shm1: {e}")

        self.shm_created = False

    def start(self):
        if self.running:
            print("Already running")
            return

        print(f"Starting Tactile Sensor Publisher ({self.sensor_type})...")

        try:
            self._init_shared_memory()
            self.sensor.initialize()

            self.running = True
            print("Tactile publisher started successfully!")
            print("Shared memory blocks 'tactile_s0_shm' and 'tactile_s1_shm' are now being updated")
            print("Press Ctrl+C to stop")

        except Exception as e:
            print(f"Failed to start publisher: {e}")
            self.stop()
            raise

    def stop(self):
        if self.cleanup_done:
            return

        print("Stopping tactile publisher...")

        self.running = False
        self.sensor.cleanup()
        self._cleanup_shared_memory()

        self.cleanup_done = True
        print("Tactile publisher stopped")

    def publish_loop(self):
            print_counter = 0
            print_interval = 20
            sampling_rate = self.config.get('sampling_rate', 200)
            sleep_duration = 1.0 / sampling_rate

            while self.running:
                self.sensor.read_and_update(self.shm_array0, self.shm_array1)


                print_counter += 1
                if print_counter >= print_interval:
                    print_counter = 0

                time.sleep(sleep_duration)

    def run_forever(self):
        self.start()
        try:
            # background thread: shared memory publishing
            loop_thread = threading.Thread(target=self.publish_loop, daemon=True)
            loop_thread.start()

            # If sensor has a GUI/visualizer, run it on the MAIN thread (matplotlib is happier here)
            if hasattr(self.sensor, "visualize"):
                self.sensor.visualize()
            else:
                loop_thread.join()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sensor_type = sys.argv[1]
    else:
        sensor_type = 'egain'

    print(f"Starting with sensor type: {sensor_type}")
    publisher = TactileSensorPublisher(sensor_type=sensor_type)
    publisher.run_forever()
