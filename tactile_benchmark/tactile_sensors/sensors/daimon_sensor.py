"""
Daimon tactile sensor implementation with dual-rate data streaming.
High-frequency (120 FPS): depth, deformation, shear arrays
Medium-frequency (60 FPS): 4-panel visualization composite
"""

import time
import threading
import numpy as np
import cv2
from multiprocessing import shared_memory

from .base_sensor import BaseTactileSensor
from sensor_configs import DAIMON_CONFIG

try:
    from dmrobotics import Sensor, put_arrows_on_image
except ImportError:
    print("Warning: dmrobotics not installed. Daimon sensor will not work.")
    Sensor = None
    put_arrows_on_image = None


class DaimonSensor(BaseTactileSensor):
    def __init__(self):
        self.sensor0 = None
        self.sensor1 = None

        self.running = False
        self.data_thread0 = None
        self.data_thread1 = None
        self.viz_thread = None

        self.depth0 = None
        self.deformation0 = None
        self.shear0 = None
        self.raw_img0 = None

        self.depth1 = None
        self.deformation1 = None
        self.shear1 = None
        self.raw_img1 = None

        self.data_lock0 = threading.Lock()
        self.data_lock1 = threading.Lock()

        self.shm_img0 = None
        self.shm_img1 = None
        self.shm_img_array0 = None
        self.shm_img_array1 = None
        self.shm_img_created = False

        self.windows_created = False

    def initialize(self):
        if Sensor is None:
            raise ImportError("dmrobotics library not available")

        print("Initializing Daimon sensors...")
        device_id_0 = DAIMON_CONFIG['id_0']
        device_id_1 = DAIMON_CONFIG['id_1']

        self.sensor0 = Sensor(device_id_0)
        self.sensor1 = Sensor(device_id_1)
        time.sleep(5.0)
        self.sensor0.reset()
        self.sensor1.reset()
        print("Sensors reset/calibrated")

        test_depth = self.sensor0.getDepth()
        test_deformation = self.sensor0.getDeformation2D()
        test_shear = self.sensor0.getShear()

        data_size = test_depth.size + test_deformation.size + test_shear.size
        DAIMON_CONFIG['data_shape'] = (data_size,)

        self._init_image_shared_memory()

        self.running = True

        self.data_thread0 = threading.Thread(target=self._data_loop, args=(0,), daemon=True)
        self.data_thread1 = threading.Thread(target=self._data_loop, args=(1,), daemon=True)
        self.viz_thread = threading.Thread(target=self._viz_loop, daemon=True)

        self.data_thread0.start()
        self.data_thread1.start()
        self.viz_thread.start()

        print("Daimon sensors initialized")

    def _init_image_shared_memory(self):
        h, w = DAIMON_CONFIG['img_height'], DAIMON_CONFIG['img_width']
        composite_h, composite_w = h * 2, w * 2
        img_size = composite_h * composite_w * 3

        try:
            self.shm_img0 = shared_memory.SharedMemory(
                name="tactile_s0_img_shm",
                create=True,
                size=img_size
            )
            self.shm_img1 = shared_memory.SharedMemory(
                name="tactile_s1_img_shm",
                create=True,
                size=img_size
            )
        except FileExistsError:
            print("Daimon image shared memory exists, reconnecting...")
            try:
                self.shm_img0 = shared_memory.SharedMemory(name="tactile_s0_img_shm")
                self.shm_img1 = shared_memory.SharedMemory(name="tactile_s1_img_shm")
            except Exception as e:
                print(f"Failed to reconnect: {e}. Recreating...")
                self._cleanup_image_shared_memory()
                self.shm_img0 = shared_memory.SharedMemory(
                    name="tactile_s0_img_shm",
                    create=True,
                    size=img_size
                )
                self.shm_img1 = shared_memory.SharedMemory(
                    name="tactile_s1_img_shm",
                    create=True,
                    size=img_size
                )

        self.shm_img_array0 = np.ndarray((composite_h, composite_w, 3), dtype=np.uint8, buffer=self.shm_img0.buf)
        self.shm_img_array1 = np.ndarray((composite_h, composite_w, 3), dtype=np.uint8, buffer=self.shm_img1.buf)

        self.shm_img_array0.fill(0)
        self.shm_img_array1.fill(0)

        self.shm_img_created = True
        print(f"Daimon image shared memory initialized ({img_size} bytes each)")

    def _cleanup_image_shared_memory(self):
        if not self.shm_img_created:
            return

        try:
            if self.shm_img0:
                self.shm_img0.close()
                try:
                    self.shm_img0.unlink()
                except FileNotFoundError:
                    pass
                self.shm_img0 = None
        except Exception as e:
            print(f"Error cleaning up daimon shm_img0: {e}")

        try:
            if self.shm_img1:
                self.shm_img1.close()
                try:
                    self.shm_img1.unlink()
                except FileNotFoundError:
                    pass
                self.shm_img1 = None
        except Exception as e:
            print(f"Error cleaning up daimon shm_img1: {e}")

        self.shm_img_created = False

    def _data_loop(self, sensor_id):
        """High-frequency loop: 120 FPS for depth, deformation, shear"""
        sensor = self.sensor0 if sensor_id == 0 else self.sensor1
        lock = self.data_lock0 if sensor_id == 0 else self.data_lock1
        sleep_time = 1.0 / DAIMON_CONFIG['data_sampling_rate']

        while self.running:
            start = time.time()

            depth = sensor.getDepth()
            deformation = sensor.getDeformation2D()
            shear = sensor.getShear()
            raw_img = sensor.getRawImage()

            with lock:
                if sensor_id == 0:
                    self.depth0 = depth
                    self.deformation0 = deformation
                    self.shear0 = shear
                    self.raw_img0 = raw_img
                else:
                    self.depth1 = depth
                    self.deformation1 = deformation
                    self.shear1 = shear
                    self.raw_img1 = raw_img

            elapsed = time.time() - start
            sleep_duration = max(0, sleep_time - elapsed)
            time.sleep(sleep_duration)

    def _viz_loop(self):
        """Medium-frequency loop: 60 FPS for visualization composite"""
        sleep_time = 1.0 / DAIMON_CONFIG['viz_sampling_rate']
        h, w = DAIMON_CONFIG['img_height'], DAIMON_CONFIG['img_width']

        while self.running:
            start = time.time()

            if self.raw_img0 is not None and self.raw_img1 is not None:
                with self.data_lock0:
                    raw0, depth0, deform0, shear0 = self.raw_img0, self.depth0, self.deformation0, self.shear0
                with self.data_lock1:
                    raw1, depth1, deform1, shear1 = self.raw_img1, self.depth1, self.deformation1, self.shear1

                composite0 = self._create_composite(raw0, depth0, deform0, shear0, h, w)
                composite1 = self._create_composite(raw1, depth1, deform1, shear1, h, w)

                np.copyto(self.shm_img_array0, composite0)
                np.copyto(self.shm_img_array1, composite1)

            elapsed = time.time() - start
            sleep_duration = max(0, sleep_time - elapsed)
            time.sleep(sleep_duration)

    def _create_composite(self, raw_img, depth, deformation, shear, h, w):
        """Create 2x2 composite visualization"""
        raw_viz = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR) if len(raw_img.shape) == 2 else raw_img
        raw_viz = cv2.resize(raw_viz, (w, h))

        depth_viz = cv2.applyColorMap((depth * 0.25 * 255.0).astype('uint8'), cv2.COLORMAP_HOT)
        depth_viz = cv2.resize(depth_viz, (w, h))

        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        deform_viz = put_arrows_on_image(black_img.copy(), deformation * 20)
        shear_viz = put_arrows_on_image(black_img.copy(), shear * 20)

        top_row = np.hstack([raw_viz, depth_viz])
        bottom_row = np.hstack([deform_viz, shear_viz])
        composite = np.vstack([top_row, bottom_row])

        return composite

    def read_and_update(self, shm_array0, shm_array1):
        """Update shared memory with high-frequency tactile data"""
        if not self.windows_created:
            cv2.namedWindow("Daimon Sensor 0", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Daimon Sensor 1", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Daimon Sensor 0", 640, 480)
            cv2.resizeWindow("Daimon Sensor 1", 640, 480)
            self.windows_created = True

        with self.data_lock0:
            if self.depth0 is not None:
                data0 = np.concatenate([
                    self.depth0.flatten(),
                    self.deformation0.flatten(),
                    self.shear0.flatten()
                ])
                shm_array0[:] = data0

        with self.data_lock1:
            if self.depth1 is not None:
                data1 = np.concatenate([
                    self.depth1.flatten(),
                    self.deformation1.flatten(),
                    self.shear1.flatten()
                ])
                shm_array1[:] = data1

        if self.shm_img_created:
            composite0_rgb = cv2.cvtColor(self.shm_img_array0, cv2.COLOR_RGB2BGR)
            composite1_rgb = cv2.cvtColor(self.shm_img_array1, cv2.COLOR_RGB2BGR)

            cv2.imshow("Daimon Sensor 0", composite0_rgb)
            cv2.imshow("Daimon Sensor 1", composite1_rgb)
            cv2.waitKey(1)

    def cleanup(self):
        self.running = False

        if self.data_thread0:
            self.data_thread0.join(timeout=1.0)
        if self.data_thread1:
            self.data_thread1.join(timeout=1.0)
        if self.viz_thread:
            self.viz_thread.join(timeout=1.0)

        if self.sensor0:
            self.sensor0.disconnect()
        if self.sensor1:
            self.sensor1.disconnect()

        self._cleanup_image_shared_memory()
        cv2.destroyAllWindows()

        print("Daimon sensors closed")
