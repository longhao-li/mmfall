import numpy as np
import json
from typing import List


UART_MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

MMWDEMO_OUTPUT_MSG_DETECTED_POINTS                  = 1
MMWDEMO_OUTPUT_MSG_RANGE_PROFILE                    = 2
MMWDEMO_OUTPUT_MSG_NOISE_PROFILE                    = 3
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP           = 4
MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP           = 5
MMWDEMO_OUTPUT_MSG_STATS                            = 6
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO        = 7
MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP = 8
MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS                = 9
MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS                 = 1000
MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST       = 1010
MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX         = 1011
MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT        = 1012
MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS                = 1020
MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION             = 1021
MMWDEMO_OUTPUT_MSG_OCCUPANCY_STATE_MACHINE          = 1030
MMWDEMO_OUTPUT_MSG_GESTURE_FEATURES                 = 1050
MMWDEMO_OUTPUT_MSG_ANN_OP_PROB                      = 1051


class TLV:
    def __init__(self) -> None:
        self.type: int   = 0
        self.length: int = 0

    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}}}'
    
    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length

    def parse(self, tlv_data: bytes) -> int:
        self.type   = int.from_bytes(tlv_data[0:4], byteorder='little', signed=False)
        self.length = int.from_bytes(tlv_data[4:8], byteorder='little', signed=False)
        return self.length + 8


class PointCloud(TLV):
    def __init__(self) -> None:
        super(PointCloud, self).__init__()
        # x: m, y: m, z: m, doppler: m/s
        self.points: None | np.ndarray = None

    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}, "point_cloud": {self.points}}}'
    
    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length
        yield 'point_cloud', self.points

    def parse(self, tlv_data: bytes) -> int:
        super().parse(tlv_data)
        assert self.type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS, f"TLV type is expected to be {MMWDEMO_OUTPUT_MSG_DETECTED_POINTS}, but got {self.type}"
        assert self.length % 16 == 0
        self.points = np.frombuffer(tlv_data[8:self.length + 8], dtype=np.float32).reshape(-1, 4)
        return self.length + 8


class RangeProfile(TLV):
    def __init__(self) -> None:
        super(RangeProfile, self).__init__()
        # sum of log2 magnitudes of received antennas
        self.points: None | np.ndarray = None

    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}, "range_profile": {self.points}}}'

    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length
        yield 'range_profile', self.points

    def parse(self, tlv_data: bytes) -> int:
        super().parse(tlv_data)
        assert self.type == MMWDEMO_OUTPUT_MSG_RANGE_PROFILE, f"TLV type is expected to be {MMWDEMO_OUTPUT_MSG_RANGE_PROFILE}, but got {self.type}"
        assert self.length % 2 == 0
        q9_doppler   = np.frombuffer(tlv_data[8:self.length + 8], dtype=np.uint16)
        decimals     = (q9_doppler & 0x1FF).astype(np.float32) / 512.0
        integers     = ((q9_doppler >> 9) & 0x3F).astype(np.float32)
        signed       = (-(q9_doppler >> 14) + 1).astype(np.float32)
        self.points  = (integers + decimals) * signed
        return self.length + 8


class NoiseProfile(TLV):
    def __init__(self) -> None:
        super(NoiseProfile, self).__init__()
        self.points: None | np.ndarray = None

    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}, "noise_profile": {self.points}}}'
    
    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length
        yield 'noise_profile', self.points

    def parse(self, tlv_data: bytes) -> int:
        super().parse(tlv_data)
        assert self.type == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE, f"TLV type is expected to be {MMWDEMO_OUTPUT_MSG_NOISE_PROFILE}, but got {self.type}"
        assert self.length % 2 == 0
        q9_doppler   = np.frombuffer(tlv_data[8:self.length + 8], dtype=np.uint16)
        decimals     = (q9_doppler & 0x1FF).astype(np.float32) / 512.0
        integers     = ((q9_doppler >> 9) & 0x3F).astype(np.float32)
        signed       = (-(q9_doppler >> 14) + 1).astype(np.float32)
        self.points  = (integers + decimals) * signed
        return self.length + 8


class Statistics(TLV):
    def __init__(self) -> None:
        super(Statistics, self).__init__()
        self.interframe_processing_time   = 0   # usec
        self.transmit_output_time         = 0   # usec
        self.interframe_processing_margin = 0   # usec
        self.interchirp_processing_margin = 0   # usec
        self.activeframe_cpu_load         = 0   # %
        self.interframe_cpu_load          = 0   # %

    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}, "interframe_processing_time": {self.interframe_processing_time}, "transmit_output_time": {self.transmit_output_time}, "interframe_processing_margin": {self.interframe_processing_margin}, "interchirp_processing_margin": {self.interchirp_processing_margin}, "activeframe_cpu_load": {self.activeframe_cpu_load}, "interframe_cpu_load": {self.interframe_cpu_load}}}'

    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length
        yield 'interframe_processing_time', self.interframe_processing_time
        yield 'transmit_output_time', self.transmit_output_time
        yield 'interframe_processing_margin', self.interframe_processing_margin
        yield 'interchirp_processing_margin', self.interchirp_processing_margin
        yield 'activeframe_cpu_load', self.activeframe_cpu_load
        yield 'interframe_cpu_load', self.interframe_cpu_load

    def parse(self, tlv_data: bytes) -> int:
        super().parse(tlv_data)
        assert self.type == MMWDEMO_OUTPUT_MSG_STATS, f"TLV type is expected to be {MMWDEMO_OUTPUT_MSG_STATS}, but got {self.type}"
        assert self.length == 24
        self.interframe_processing_time   = int.from_bytes(tlv_data[8:12], byteorder='little', signed=False)
        self.transmit_output_time         = int.from_bytes(tlv_data[12:16], byteorder='little', signed=False)
        self.interframe_processing_margin = int.from_bytes(tlv_data[16:20], byteorder='little', signed=False)
        self.interchirp_processing_margin = int.from_bytes(tlv_data[20:24], byteorder='little', signed=False)
        self.activeframe_cpu_load         = int.from_bytes(tlv_data[24:28], byteorder='little', signed=False)
        self.interframe_cpu_load          = int.from_bytes(tlv_data[28:32], byteorder='little', signed=False)
        return self.length + 8


class SideInfoForDetectedPoints(TLV):
    def __init__(self) -> None:
        super(SideInfoForDetectedPoints, self).__init__()
        self.SNR: None | np.ndarray   = None    # dB
        self.noise: None | np.ndarray = None    # dB
    
    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}, "SNR": {self.SNR}, "noise": {self.noise}}}'

    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length
        yield 'SNR', self.SNR
        yield 'noise', self.noise

    def parse(self, tlv_data: bytes) -> int:
        super().parse(tlv_data)
        assert self.type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO, f"TLV type is expected to be {MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO}, but got {self.type}"
        assert self.length % 4 == 0
        mixed_data = np.frombuffer(tlv_data[8:self.length + 8], dtype=np.uint16).reshape(-1, 2).astype(np.float32) * 0.1
        self.SNR   = mixed_data[:, 0]
        self.noise = mixed_data[:, 1]
        return self.length + 8


class TemperatureStatistics(TLV):
    def __init__(self) -> None:
        super(TemperatureStatistics, self).__init__()
        self.is_valid         = False
        self.time             = 0       # ms
        self.Rx0_temperature  = 0       # degree C
        self.Rx1_temperature  = 0       # degree C
        self.Rx2_temperature  = 0       # degree C
        self.Rx3_temperature  = 0       # degree C
        self.Tx0_temperature  = 0       # degree C
        self.Tx1_temperature  = 0       # degree C
        self.Tx2_temperature  = 0       # degree C
        self.Pm_temperature   = 0       # degree C
        self.dig0_temperature = 0       # degree C
        self.dig1_temperature = 0       # degree C
    
    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}, "is_valid": {self.is_valid}, "time": {self.time}, "Rx0_temperature": {self.Rx0_temperature}, "Rx1_temperature": {self.Rx1_temperature}, "Rx2_temperature": {self.Rx2_temperature}, "Rx3_temperature": {self.Rx3_temperature}, "Tx0_temperature": {self.Tx0_temperature}, "Tx1_temperature": {self.Tx1_temperature}, "Tx2_temperature": {self.Tx2_temperature}, "Pm_temperature": {self.Pm_temperature}, "dig0_temperature": {self.dig0_temperature}, "dig1_temperature": {self.dig1_temperature}}}'

    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length
        yield 'is_valid', self.is_valid
        yield 'time', self.time
        yield 'Rx0_temperature', self.Rx0_temperature
        yield 'Rx1_temperature', self.Rx1_temperature
        yield 'Rx2_temperature', self.Rx2_temperature
        yield 'Rx3_temperature', self.Rx3_temperature
        yield 'Tx0_temperature', self.Tx0_temperature
        yield 'Tx1_temperature', self.Tx1_temperature
        yield 'Tx2_temperature', self.Tx2_temperature
        yield 'Pm_temperature', self.Pm_temperature
        yield 'dig0_temperature', self.dig0_temperature
        yield 'dig1_temperature', self.dig1_temperature

    def parse(self, tlv_data: bytes) -> int:
        super().parse(tlv_data)
        assert self.type == MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS, f"TLV type is expected to be {MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS}, but got {self.type}"
        assert self.length == 28

        self.is_valid         = int.from_bytes(tlv_data[8:12], byteorder='little', signed=False) != 0
        self.time             = int.from_bytes(tlv_data[12:16], byteorder='little', signed=False)
        self.Rx0_temperature  = int.from_bytes(tlv_data[16:18], byteorder='little', signed=True)
        self.Rx1_temperature  = int.from_bytes(tlv_data[18:20], byteorder='little', signed=True)
        self.Rx2_temperature  = int.from_bytes(tlv_data[20:22], byteorder='little', signed=True)
        self.Rx3_temperature  = int.from_bytes(tlv_data[22:24], byteorder='little', signed=True)
        self.Tx0_temperature  = int.from_bytes(tlv_data[24:26], byteorder='little', signed=True)
        self.Tx1_temperature  = int.from_bytes(tlv_data[26:28], byteorder='little', signed=True)
        self.Tx2_temperature  = int.from_bytes(tlv_data[28:30], byteorder='little', signed=True)
        self.Pm_temperature   = int.from_bytes(tlv_data[30:32], byteorder='little', signed=True)
        self.dig0_temperature = int.from_bytes(tlv_data[32:34], byteorder='little', signed=True)
        self.dig1_temperature = int.from_bytes(tlv_data[34:36], byteorder='little', signed=True)
        return self.length + 8


class SphericalPoints(TLV):
    def __init__(self) -> None:
        super(SphericalPoints, self).__init__()
        # range: m, azimuth: rad, elevation: rad, doppler: m/s
        self.points: None | np.ndarray = None
    
    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}, "spherical_points": {self.points}}}'
    
    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length
        yield 'spherical_points', self.points

    def parse(self, tlv_data: bytes) -> int:
        super().parse(tlv_data)
        assert self.type == MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS, f"TLV type is expected to be {MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS}, but got {self.type}"
        assert self.length % 16 == 0
        self.points = np.frombuffer(tlv_data[8:self.length + 8], dtype=np.float32).reshape(-1, 4)
        return self.length + 8


class TargetInfo:
    def __init__(self) -> None:
        self.track_id                                   = 0
        self.position: None | np.ndarray                = None
        self.velocity: None | np.ndarray                = None
        self.acceleration: None | np.ndarray            = None
        self.error_covariance_matrix: None | np.ndarray = None
        self.gating_function_gain                       = 0
        self.confidence_level                           = 0
    
    def __str__(self) -> str:
        return f'{{"track_id": {self.track_id}, "position": {self.position}, "velocity": {self.velocity}, "acceleration": {self.acceleration}, "error_covariance_matrix": {self.error_covariance_matrix}, "gating_function_gain": {self.gating_function_gain}, "confidence_level": {self.confidence_level}}}'
        
    def __iter__(self):
        yield 'track_id', self.track_id
        yield 'position', self.position
        yield 'velocity', self.velocity
        yield 'acceleration', self.acceleration
        yield 'error_covariance_matrix', self.error_covariance_matrix
        yield 'gating_function_gain', self.gating_function_gain
        yield 'confidence_level', self.confidence_level

    def parse(self, data: bytes) -> None:
        self.track_id                = int.from_bytes(data[0:4], byteorder='little', signed=False)
        floats                       = np.frombuffer(data[4:112], dtype=np.float32)
        self.position                = floats[0:3]
        self.velocity                = floats[3:6]
        self.acceleration            = floats[6:9]
        self.error_covariance_matrix = floats[9:25].reshape(4, 4)
        self.gating_function_gain    = floats[25]
        self.confidence_level        = floats[26]


class TargetList(TLV):
    def __init__(self) -> None:
        super(TargetList, self).__init__()
        self.targets: None | List[TargetInfo] = None
    
    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}, "targets": {self.targets}}}'
    
    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length
        yield 'targets', self.targets

    def parse(self, tlv_data: bytes) -> int:
        super().parse(tlv_data)
        assert self.type == MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST, f"TLV type is expected to be {MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST}, but got {self.type}"
        assert self.length % 112 == 0
        self.targets = list()
        for i in range(self.length // 112):
            target = TargetInfo()
            target.parse(tlv_data[8 + i * 112: 8 + (i + 1) * 112])
            self.targets.append(target)
        return self.length + 8


class TargetIndex(TLV):
    def __init__(self) -> None:
        super(TargetIndex, self).__init__()
        self.target_index: None | np.ndarray = None
    
    def __str__(self) -> str:
        return f'{{"type": {self.type}, "length": {self.length}, "target_index": {self.target_index}}}'
    
    def __iter__(self):
        yield 'type', self.type
        yield 'length', self.length
        yield 'target_index', self.target_index

    def parse(self, tlv_data: bytes) -> int:
        super().parse(tlv_data)
        assert self.type == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX, f"TLV type is expected to be {MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX}, but got {self.type}"
        self.target_index = np.frombuffer(tlv_data[8:self.length + 8], dtype=np.uint8)
        return self.length + 8
