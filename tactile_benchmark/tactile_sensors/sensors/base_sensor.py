"""
Base class for tactile sensors.
Defines the interface that all sensor subclasses must implement.
"""

from abc import ABC, abstractmethod


class BaseTactileSensor(ABC):
    """Abstract base class for tactile sensors"""

    @abstractmethod
    def initialize(self):
        """Initialize the sensor (serial connections, calibration, etc.)"""
        pass

    @abstractmethod
    def read_and_update(self, shm_array0, shm_array1):
        """
        Read sensor data and update shared memory arrays.

        Args:
            shm_array0: Numpy array backed by shared memory for sensor 0/left
            shm_array1: Numpy array backed by shared memory for sensor 1/right
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources (close serial connections, etc.)"""
        pass
