"""
Eflesh sensor implementation using AnySkin package.
Supports dual sensors (left and right).
"""

import time
import numpy as np
import cv2
import threading
from anyskin import AnySkinProcess

from .base_sensor import BaseTactileSensor
from sensor_configs import EFLESH_CONFIG


class EfleshSensor(BaseTactileSensor):
    def __init__(self):
        self.stream_left = None
        self.stream_right = None
        
        self.baseline_left = None
        self.baseline_right = None
        
        self.running = False
        self.windows_created = False
        
        # Visualization constants
        self.chip_locations = np.array([[455, 453], [275, 451], [624, 455], [451, 292], [454, 613]])
        self.chip_xy_rotations = np.array([-np.pi / 2, -np.pi / 2, np.pi, np.pi / 2, 0.0])
        self.bg_image = None
        self.viz_scaling = EFLESH_CONFIG.get('scaling', 7.0)

    def initialize(self):
        print("Initializing Eflesh sensors...")
        print(f"Left sensor on {EFLESH_CONFIG['port_left']}")
        # print(f"Right sensor on {EFLESH_CONFIG['port_right']}")
        
        try:
            # Initialize Left Sensor
            self.stream_left = AnySkinProcess(
                num_mags=EFLESH_CONFIG['num_mags'],
                port=EFLESH_CONFIG['port_left']
            )
            self.stream_left.start()
            
            # Initialize Right Sensor
            self.stream_right = AnySkinProcess(
                num_mags=EFLESH_CONFIG['num_mags'],
                port=EFLESH_CONFIG['port_right']
            )
            self.stream_right.start()
            
            time.sleep(1.0)
            
            print("Calibrating sensors...")
            self.baseline_left = self._get_baseline(self.stream_left)
            print("Left sensor calibrated")
            
            self.baseline_right = self._get_baseline(self.stream_right)
            print("Right sensor calibrated")
            
            self.running = True
            print("Eflesh sensors initialized")
            
        except Exception as e:
            print(f"Failed to initialize Eflesh sensors: {e}")
            self.cleanup()
            raise

    def _get_baseline(self, stream):
        baseline_data = stream.get_data(num_samples=5)
        print(baseline_data.shape)
        # baseline_data = np.array(baseline_data)[:, 1:]  # Skip timestamp
        baseline = np.mean(baseline_data, axis=0)
        return baseline

    def read_and_update(self, shm_array0, shm_array1):
        if not self.running:
            return

        if not self.windows_created:
            cv2.namedWindow("Eflesh Left", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Eflesh Right", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Eflesh Left", 600, 600)
            cv2.resizeWindow("Eflesh Right", 600, 600)
            
            self.bg_image = np.ones((900, 900, 3), dtype=np.uint8) * 235
            self.windows_created = True

        try:
            # Process Left Sensor
            self._process_sensor(self.stream_left, self.baseline_left, shm_array0, "Eflesh Left")
            
            # Process Right Sensor
            self._process_sensor(self.stream_right, self.baseline_right, shm_array1, "Eflesh Right")
            
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Error reading/updating Eflesh sensors: {e}")

    def _process_sensor(self, stream, baseline, shm_array, window_name):
        raw_data = stream.get_data(num_samples=1)
        if not raw_data:
            return

        sensor_data = raw_data[0][1:]
        processed_data = sensor_data - baseline
        
        # Reshape and Flip
        data_reshaped = processed_data.reshape(-1, 3)
        data_reshaped[:, :2] *= -1
        
        # Update Shared Memory
        flat_data = data_reshaped.flatten()
        shm_array[:len(flat_data)] = flat_data
        
        # Visualize
        self._visualize(data_reshaped, window_name)

    def _visualize(self, data_reshaped, window_name):
        canvas = self.bg_image.copy()
        
        for magid, chip_location in enumerate(self.chip_locations):
            # Draw circle for Z magnitude
            z_val = np.abs(data_reshaped[magid, 2])
            radius = int(z_val / self.viz_scaling)
            if radius < 1: radius = 1
            
            color = (0, 0, 255) # Red
            thickness = 2 if data_reshaped[magid, 2] < 0 else -1
            
            cv2.circle(canvas, tuple(chip_location), radius, color, thickness)
            
            # Draw vector
            arrow_start = tuple(chip_location)
            rot = self.chip_xy_rotations[magid]
            rotation_mat = np.array([
                [np.cos(rot), -np.sin(rot)],
                [np.sin(rot), np.cos(rot)]
            ])
            
            xy_data = data_reshaped[magid, :2].copy()
            if data_reshaped[magid, 2] < 0:
                xy_data *= -1
                
            data_xy = np.dot(rotation_mat, xy_data)
            
            arrow_end = (
                int(chip_location[0] + data_xy[0] / self.viz_scaling),
                int(chip_location[1] + data_xy[1] / self.viz_scaling)
            )
            
            cv2.arrowedLine(canvas, arrow_start, arrow_end, (0, 255, 0), 4)

        cv2.imshow(window_name, canvas)

    def cleanup(self):
        self.running = False
        
        if self.stream_left:
            print("Stopping Left Eflesh stream...")
            self.stream_left.pause_streaming()
            self.stream_left.join()
            
        if self.stream_right:
            print("Stopping Right Eflesh stream...")
            self.stream_right.pause_streaming()
            self.stream_right.join()
        
        cv2.destroyAllWindows()
        print("Eflesh sensors closed")
