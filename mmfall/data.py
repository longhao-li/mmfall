import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import BinaryIO, List, Callable


_MAGIC_WORD: bytes = b'\x02\x01\x04\x03\x06\x05\x08\x07'

_DETECT_POINTS: int                     = 1
_RANGE_PROFILE: int                     = 2
_NOISE_PROFILE: int                     = 3
_AZIMUTH_STATIC_HEAT_MAP: int           = 4
_RANGE_DOPPLER_HEAT_MAP: int            = 5
_STATISTICS: int                        = 6
_DETECT_POINTS_SIDE_INFO: int           = 7
_AZIMUTH_ELEVATION_STATIC_HEAT_MAP: int = 8
_TEMPERATURE_STATISTICS: int            = 9
_SPHERICAL_POINTS: int                  = 1000
_3D_TARGET_LIST: int                    = 1010
_TARGET_INDEX: int                      = 1011
_TARGET_HEIGHT: int                     = 1012
_COMPRESSED_SPHERICAL_POINTS: int       = 1020
_PRESENCE_INDICATION: int               = 1021
_OCCUPANCY_STATE_MACHINE: int           = 1030
_GESTURE_FEATURES: int                  = 1050
_ANN_OP_PROB: int                       = 1051


def _parse_detected_points(tlv: bytes) -> Tensor:
    tlv_type   = int.from_bytes(tlv[0:4], 'little', signed=False)
    tlv_length = int.from_bytes(tlv[4:8], 'little', signed=False)
    assert tlv_type == _DETECT_POINTS

    content    = tlv[8:tlv_length + 8]
    num_points = tlv_length // 16

    points = np.frombuffer(content, dtype=np.float32).reshape((num_points, 4))
    return torch.from_numpy(points)
    


def _parse_spherical_points(tlv: bytes) -> Tensor:
    tlv_type   = int.from_bytes(tlv[0:4], 'little', signed=False)
    tlv_length = int.from_bytes(tlv[4:8], 'little', signed=False)
    assert tlv_type == _SPHERICAL_POINTS
    
    content    = tlv[8:tlv_length + 8]
    num_points = tlv_length // 16

    spherical = np.frombuffer(content, dtype=np.float32).reshape((num_points, 4))
    
    x       = spherical[:, 0] * np.sin(spherical[:, 1]) * np.cos(spherical[:, 2])
    y       = spherical[:, 0] * np.cos(spherical[:, 1]) * np.cos(spherical[:, 2])
    z       = spherical[:, 0] * np.sin(spherical[:, 2])
    doppler = spherical[:, 3]

    points = np.stack((x, y, z, doppler), axis=1)
    return torch.from_numpy(points)


def _parse_compressed_spherical_points(tlv: bytes) -> Tensor:
    tlv_type   = int.from_bytes(tlv[0:4], 'little', signed=False)
    tlv_length = int.from_bytes(tlv[4:8], 'little', signed=False)
    assert tlv_type == _COMPRESSED_SPHERICAL_POINTS

    content = tlv[8:tlv_length + 8]
    elevation_unit, azimuth_unit, doppler_unit, range_unit, snr_unit = np.frombuffer(content[0:20], dtype=np.float32)

    content    = content[20:]
    num_points = tlv_length // 8
    points     = np.empty((num_points, 5), dtype=np.float32)

    for i in range(num_points):
        point_data = content[i * 8:(i + 1) * 8]

        elevation = int.from_bytes(point_data[0:1], 'little', signed=True) * elevation_unit
        azimuth   = int.from_bytes(point_data[1:2], 'little', signed=True) * azimuth_unit
        doppler   = int.from_bytes(point_data[2:4], 'little', signed=True) * doppler_unit
        distance  = int.from_bytes(point_data[4:6], 'little', signed=False) * range_unit
        snr       = int.from_bytes(point_data[6:8], 'little', signed=False) * snr_unit

        x = distance * np.sin(azimuth) * np.cos(elevation)
        y = distance * np.cos(azimuth) * np.cos(elevation)
        z = distance * np.sin(elevation)

        points[i, 0] = x
        points[i, 1] = y
        points[i, 2] = z
        points[i, 3] = doppler
        points[i, 4] = snr
    
    return torch.from_numpy(points)


def parse_frame(frame: bytes) -> Tensor | None:
    # check magic word
    if len(frame) < 40 or frame[0:8] != _MAGIC_WORD:
        raise ValueError('Invalid magic word. Frame data may be corrupted.')

    # parse frame header
    length      = int.from_bytes(frame[12:16], 'little', signed=False)
    num_tlvs    = int.from_bytes(frame[32:36], 'little', signed=False)

    # check frame length
    if len(frame) < length:
        raise ValueError('Invalid frame length. Frame data may be corrupted.')

    tlv_data = frame[40:length]
    for _ in range(num_tlvs):
        tlv_type   = int.from_bytes(tlv_data[0:4], 'little', signed=False)
        tlv_length = int.from_bytes(tlv_data[4:8], 'little', signed=False)
        
        if tlv_type == _DETECT_POINTS:
            return _parse_detected_points(tlv_data[0:tlv_length + 8])
        elif tlv_type == _SPHERICAL_POINTS:
            return _parse_spherical_points(tlv_data[0:tlv_length + 8])
        elif tlv_type == _COMPRESSED_SPHERICAL_POINTS:
            return _parse_compressed_spherical_points(tlv_data[0:tlv_length + 8])

        tlv_data = tlv_data[tlv_length + 8:]
    
    return None


def parse_file(file: str | BinaryIO) -> List[Tensor] | None:
    data: None | bytes = None
    if isinstance(file, str):
        with open(file, 'rb') as f:
            data = f.read()
    elif isinstance(file, BinaryIO):
        data = file.read()
    else:
        raise TypeError('Invalid file type. String or a file object is expected.')
    
    point_dim = None
    frames = list()
    while len(data) > 0:
        header = data.find(_MAGIC_WORD)
        if header == -1:
            break

        data = data[header:]
        parsed_data = parse_frame(data)
        if parsed_data is None:
            if point_dim is not None:
                frames.append(torch.empty((0, point_dim)))
        else:
            point_dim = parsed_data.size(-1)
            frames.append(parsed_data)

        length = int.from_bytes(data[12:16], 'little', signed=False)
        data   = data[length:]
    
    if len(frames) == 0:
        return None
    else:
        return frames # May need padding.


def pad_zeros(tensor: Tensor, max_len: int = 64) -> Tensor:
    dim = tensor.size(-1)
    assert tensor.size(0) <= max_len
    pad_size = max_len - tensor.size(0)
    return torch.cat((tensor, torch.zeros(pad_size, dim)), dim=0)


class MMFallDataset(Dataset):
    def __init__(
        self,
        file: str | BinaryIO,
        pattern_size: int,
        device: torch.device = torch.device('cpu'),
        transform: Callable[[Tensor], Tensor] = pad_zeros
    ) -> None:
        if pattern_size < 1:
            raise ValueError('Invalid pattern size. Pattern size must be greater than 0.')

        self._frames       = parse_file(file)
        self._transform    = transform
        self._pattern_size = pattern_size

        for i in range(len(self._frames)):
            mean            = torch.mean(self._frames[i], dim=0)
            mean[-1]        = 0
            if mean.size(-1) == 5:
                mean[-2] = 0
            self._frames[i] = self._frames[i] - mean
            self._frames[i] = self._transform(self._frames[i].to(device))
        
        self._frames = torch.stack(self._frames, dim=0).to(device)
    
    def __len__(self) -> int:
        length = len(self._frames)
        if length < self._pattern_size:
            return 0
        return length - self._pattern_size + 1
    
    def __getitem__(self, idx: int) -> Tensor:
        length = len(self)
        idx = (idx % length) + length % length

        return self._frames[idx:idx + self._pattern_size]
