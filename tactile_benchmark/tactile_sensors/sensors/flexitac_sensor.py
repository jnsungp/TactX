"""
FlexiTac sensor implementation.
Reads from two serial ports (one per sensor), processes 12x32 grids, and visualizes data.
"""

import time
import serial
import threading
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from .base_sensor import BaseTactileSensor
from sensor_configs import FLEXITAC_CONFIG


class FlexiTacSensor(BaseTactileSensor):
    def __init__(self):
        self.ser_left = None
        self.ser_right = None

        self.contact_data_left = np.zeros((12, 32), dtype=np.float32)
        self.contact_data_right = np.zeros((12, 32), dtype=np.float32)

        self.median_left = None
        self.median_right = None

        self.prev_frame_left = np.zeros((12, 32), dtype=np.float32)
        self.prev_frame_right = np.zeros((12, 32), dtype=np.float32)

        self.running = False
        self.calibrated_left = False
        self.calibrated_right = False

        self.thread_left = None
        self.thread_right = None
        self.windows_created = False

    def initialize(self):
        print(f"Initializing FlexiTac sensors...")
        print(f"Left sensor on {FLEXITAC_CONFIG['port_left']}")
        print(f"Right sensor on {FLEXITAC_CONFIG['port_right']}")

        self.ser_left = serial.Serial(FLEXITAC_CONFIG['port_left'], FLEXITAC_CONFIG['baudrate'])
        self.ser_right = serial.Serial(FLEXITAC_CONFIG['port_right'], FLEXITAC_CONFIG['baudrate'])
        time.sleep(1)

        self.ser_left.flush()
        self.ser_right.flush()

        print("Calibrating left sensor...")
        self.median_left = self._calibrate(self.ser_left)
        self.calibrated_left = True
        print("Left sensor calibrated")

        print("Calibrating right sensor...")
        self.median_right = self._calibrate(self.ser_right)
        self.calibrated_right = True
        print("Right sensor calibrated")

        self.running = True

        self.thread_left = threading.Thread(target=self._read_thread, args=(self.ser_left, 'left'), daemon=True)
        self.thread_right = threading.Thread(target=self._read_thread, args=(self.ser_right, 'right'), daemon=True)

        self.thread_left.start()
        self.thread_right.start()

        print("FlexiTac sensors initialized")

    def _calibrate(self, ser):
        data_tac = []
        current = None

        while len(data_tac) < FLEXITAC_CONFIG['calibration_frames']:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                except:
                    line = ""

                if len(line) < 10:
                    if current is not None and len(current) == 12:
                        backup = np.array(current, dtype=np.float32)
                        data_tac.append(backup)
                    current = []
                    continue

                if current is not None:
                    str_values = line.split()
                    try:
                        int_values = [int(val) for val in str_values]
                        if len(int_values) == 32:
                            current.append(int_values)
                    except:
                        pass

        data_tac = np.array(data_tac)
        return np.median(data_tac, axis=0)

    def _read_thread(self, ser, side):
        current = None
        backup = None

        if side == 'left':
            median = self.median_left
        else:
            median = self.median_right

        while self.running:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                except:
                    line = ""

                if len(line) < 10:
                    if current is not None and len(current) == 12:
                        row_lengths = [len(row) for row in current]
                        if all(length == 32 for length in row_lengths):
                            backup = np.array(current, dtype=np.float32)
                    current = []

                    if backup is not None:
                        contact_data = backup - median - FLEXITAC_CONFIG['threshold']
                        contact_data = np.clip(contact_data, 0, 100)

                        if np.max(contact_data) < FLEXITAC_CONFIG['threshold']:
                            contact_data_norm = contact_data / FLEXITAC_CONFIG['noise_scale']
                        else:
                            contact_data_norm = contact_data / np.max(contact_data)

                        if side == 'left':
                            self.contact_data_left = contact_data_norm
                        else:
                            self.contact_data_right = contact_data_norm

                    continue

                if current is not None:
                    str_values = line.split()
                    try:
                        int_values = [int(val) for val in str_values]
                        if len(int_values) == 32:
                            current.append(int_values)
                    except:
                        pass

    def read_and_update(self, shm_array0, shm_array1):
        if not self.windows_created:
            WINDOW_WIDTH = 32 * 30
            WINDOW_HEIGHT = 12 * 30
            cv2.namedWindow("FlexiTac Left", cv2.WINDOW_NORMAL)
            cv2.namedWindow("FlexiTac Right", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("FlexiTac Left", WINDOW_WIDTH, WINDOW_HEIGHT)
            cv2.resizeWindow("FlexiTac Right", WINDOW_WIDTH, WINDOW_HEIGHT)
            self.windows_created = True

        temp_left = self._temporal_filter(self.contact_data_left, self.prev_frame_left)
        temp_right = self._temporal_filter(self.contact_data_right, self.prev_frame_right)

        self.prev_frame_left = temp_left
        self.prev_frame_right = temp_right

        shm_array0[:] = temp_left.flatten()
        shm_array1[:] = temp_right.flatten()

        left_scaled = (temp_left * 255).astype(np.uint8)
        right_scaled = (temp_right * 255).astype(np.uint8)

        colormap_left = cv2.applyColorMap(left_scaled, cv2.COLORMAP_VIRIDIS)
        colormap_right = cv2.applyColorMap(right_scaled, cv2.COLORMAP_VIRIDIS)

        cv2.imshow("FlexiTac Left", colormap_left)
        cv2.imshow("FlexiTac Right", colormap_right)
        cv2.waitKey(1)

    def _temporal_filter(self, new_frame, prev_frame):
        alpha = FLEXITAC_CONFIG['temporal_alpha']
        return alpha * new_frame + (1 - alpha) * prev_frame

    def cleanup(self):
        self.running = False

        if self.thread_left:
            self.thread_left.join(timeout=1.0)
        if self.thread_right:
            self.thread_right.join(timeout=1.0)

        if self.ser_left and self.ser_left.is_open:
            self.ser_left.close()
        if self.ser_right and self.ser_right.is_open:
            self.ser_right.close()

        cv2.destroyAllWindows()
        print("FlexiTac sensors closed")
