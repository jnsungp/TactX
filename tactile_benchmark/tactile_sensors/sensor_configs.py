"""
Sensor configuration dictionaries for different tactile sensor types.
Each sensor type has its own configuration with the parameters it needs.
"""

EGAIN_CONFIG = {
    'port': '/dev/ttyACM0',
    # 'port': 'COM14',
    'baudrate': 115200,
    'data_shape': (1,6),
    'dtype': 'int32',
   'sampling_rate': 80
}

FSR_CONFIG = {
    'port': '/dev/ttyACM0',
    'baudrate': 115200,
    'data_shape': (1,),
    'dtype': 'float32',
    'sampling_rate': 200
}

FLEXITAC_CONFIG = {
    'port_left': '/dev/ttyUSB0',
    'port_right': '/dev/ttyUSB3',
    'baudrate': 1000000,
    'data_shape': (12, 32),
    'dtype': 'float32',
    'calibration_frames': 30,
    'threshold': 12,
    'noise_scale': 60,
    'gaussian_sigma': 0.1,
    'temporal_alpha': 0.2,
    'sampling_rate': 100
}

DAIMON_CONFIG = {
    'id_0': '2505150091',
    'id_1': '2505150094',
    'img_height': 240,
    'img_width': 320,
    'data_sampling_rate': 120,
    'viz_sampling_rate': 60,
    'data_shape': (240, 320, 5),  # height × width × channels (depth + deformation_x + deformation_y + shear_x + shear_y)
    'sampling_rate': 120
}

FABRIC_CONFIG = {
    'port': '/dev/ttyACM0',
    'baudrate': 921600,
    'data_shape': (6, 8),
    'dtype': 'float32',
    'sampling_rate': 100,
    'max_pressure': 80
}

EFLESH_CONFIG = {
    'port_left': '/dev/ttyACM1',
    'port_right': '/dev/ttyACM0',
    'num_mags': 5,
    'baudrate': 115200,  # Not strictly used by AnySkinProcess but good for reference
    'data_shape': (5, 3),  # 5 magnetometers, 3 axes (x, y, z)
    'dtype': 'float32',
    'sampling_rate': 60,
    'scaling': 7.0,
    'viz_mode': '3axis'
}

def get_sensor_map():
    from sensors.fsr_sensor import FSRSensor
    from sensors.flexitac_sensor import FlexiTacSensor
    from sensors.daimon_sensor import DaimonSensor
    from sensors.eflesh_sensor import EfleshSensor
    from sensors.fabric_sensor import FabricSensor

    return {
        'fsr': (FSRSensor, FSR_CONFIG),
        'flexitac': (FlexiTacSensor, FLEXITAC_CONFIG),
        'daimon': (DaimonSensor, DAIMON_CONFIG),
        'eflesh': (EfleshSensor, EFLESH_CONFIG),
        'sensor_fabric': (FabricSensor, FABRIC_CONFIG),
    }

SENSOR_MAP = get_sensor_map()
