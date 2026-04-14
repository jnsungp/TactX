"""
FSR (Force Sensitive Resistor) sensor implementation.
Reads from a single serial port that provides data for both sensors (S0 and S1).
"""

import time
import serial
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
from status_display import update_line

from .base_sensor import BaseTactileSensor
from sensor_configs import FSR_CONFIG


class FSRSensor(BaseTactileSensor):
    def __init__(self):
        self.ser = None
        self.latest_s0 = 0
        self.latest_s1 = 0

    def initialize(self):
        print(f"Initializing FSR sensor on {FSR_CONFIG['port']}...")
        self.ser = serial.Serial(FSR_CONFIG['port'], FSR_CONFIG['baudrate'], timeout=1)
        time.sleep(2)
        print(f"FSR sensor connected at {FSR_CONFIG['baudrate']} baud")

    def read_and_update(self, shm_array0, shm_array1):
        while self.ser.in_waiting > 0:
            tactile_data = self._read_single_value()

            if tactile_data is not None:
                sensor_id = tactile_data[0]
                value = tactile_data[1]

                if sensor_id == "S0":
                    shm_array0[0] = float(value)
                    self.latest_s0 = value
                elif sensor_id == "S1":
                    shm_array1[0] = float(value)
                    self.latest_s1 = value

        update_line(2, f"Tactile: S0: {self.latest_s0:4d} | S1: {self.latest_s1:4d}")

    def _read_single_value(self):
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    data = line.split(',')
                    if len(data) >= 2:
                        sensor_id = data[0]
                        value = int(data[1])
                        return [sensor_id, value]
            except Exception as e:
                print(f"\nFSR reading/decoding error: {e}")
        return None

    def cleanup(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("FSR serial connection closed")
