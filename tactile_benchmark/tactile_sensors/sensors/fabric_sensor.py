"""
Tactile Fabric (Gripper) sensor implementation.
Single serial port streaming 260-byte binary packets.
Two pads: S0 and S1, each a 6×8 grid of uint8 pressure values.
"""

import serial
import numpy as np
import cv2

from .base_sensor import BaseTactileSensor
from sensor_configs import FABRIC_CONFIG

_DELIMITER = bytes([0xAA, 0x55, 0x03, 0x99])
_PACKET_SIZE = 260  # 4-byte delimiter + 256-byte payload

# S0[r, c] at 0-indexed payload position: 10 + c*16 + r
_S0_INDICES = [10 + c * 16 + r for r in range(6) for c in range(8)]
# S1[r, c] at 0-indexed payload position: 244 - c*16 + r
_S1_INDICES = [244 - c * 16 + r for r in range(6) for c in range(8)]


class FabricSensor(BaseTactileSensor):
    def __init__(self):
        self.ser = None
        self._buf = bytearray()
        self._s0 = np.zeros(48, dtype=np.float32)
        self._s1 = np.zeros(48, dtype=np.float32)
        self._windows_created = False

    def initialize(self):
        print(f"Initializing FabricSensor on {FABRIC_CONFIG['port']}...")
        self.ser = serial.Serial(FABRIC_CONFIG['port'], FABRIC_CONFIG['baudrate'], timeout=0)
        self.ser.reset_input_buffer()
        print("FabricSensor initialized")

    def read_and_update(self, shm_array0, shm_array1):
        if not self._windows_created:
            cv2.namedWindow("Fabric S0", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Fabric S1", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Fabric S0", 8 * 180, 6 * 60)
            cv2.resizeWindow("Fabric S1", 8 * 180, 6 * 60)
            self._windows_created = True

        if self.ser.in_waiting:
            self._buf.extend(self.ser.read(self.ser.in_waiting))

        while True:
            idx = self._buf.find(_DELIMITER)
            if idx == -1:
                if len(self._buf) > _PACKET_SIZE:
                    self._buf = self._buf[-(len(_DELIMITER) - 1):]
                break
            if idx > 0:
                self._buf = self._buf[idx:]
            if len(self._buf) < _PACKET_SIZE:
                break
            payload = self._buf[4:_PACKET_SIZE]
            self._buf = self._buf[_PACKET_SIZE:]
            self._s0[:] = [payload[i] for i in _S0_INDICES]
            self._s1[:] = [payload[i] for i in _S1_INDICES]

        shm_array0[:] = self._s0
        shm_array1[:] = self._s1

        s0_norm = np.clip(self._s0.reshape(6, 8) / FABRIC_CONFIG['max_pressure'], 0, 1)
        s1_norm = np.clip(self._s1.reshape(6, 8) / FABRIC_CONFIG['max_pressure'], 0, 1)
        s0_img = cv2.resize(cv2.applyColorMap((s0_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), (8 * 180, 6 * 60), interpolation=cv2.INTER_NEAREST)
        s1_img = cv2.resize(cv2.applyColorMap((s1_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), (8 * 180, 6 * 60), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Fabric S0", s0_img)
        cv2.imshow("Fabric S1", s1_img)
        cv2.waitKey(1)

    def cleanup(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        cv2.destroyAllWindows()
        print("FabricSensor closed")
