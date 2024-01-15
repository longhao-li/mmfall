import numpy as np
import mmfall.data as data
import sys
import json
from typing import Dict, List


def parse_frame(frame: bytes) -> Dict:
    assert len(frame) >= 40, "Invalid frame length. Frame may be corrupted."
    assert frame[:8] == data.UART_MAGIC_WORD, "Invalid frame header. Frame may be corrupted."

    frame_id         = int.from_bytes(frame[20:24], byteorder = 'little', signed = False)
    timestamp        = int.from_bytes(frame[24:28], byteorder = 'little', signed = False)
    num_detected_obj = int.from_bytes(frame[28:32], byteorder = 'little', signed = False)
    num_tlvs         = int.from_bytes(frame[32:36], byteorder = 'little', signed = False)

    frame_data = dict()

    frame_data['frame_id']         = frame_id
    frame_data['timestamp']        = timestamp
    frame_data['num_detected_obj'] = num_detected_obj
    frame_data['num_tlvs']         = num_tlvs
    frame_data['tlvs']             = list()

    tlv_data = frame[40:]
    for _ in range(num_tlvs):
        tlv_type   = int.from_bytes(tlv_data[0:4], byteorder = 'little', signed = False)
        tlv_length = int.from_bytes(tlv_data[4:8], byteorder = 'little', signed = False)

        if tlv_type == data.MMWDEMO_OUTPUT_MSG_DETECTED_POINTS:
            point_cloud = data.PointCloud()
            point_cloud.parse(tlv_data)
            frame_data['tlvs'].append(point_cloud)
        elif tlv_type == data.MMWDEMO_OUTPUT_MSG_RANGE_PROFILE:
            profile = data.RangeProfile()
            profile.parse(tlv_data)
            frame_data['tlvs'].append(profile)
        elif tlv_type == data.MMWDEMO_OUTPUT_MSG_NOISE_PROFILE:
            profile = data.NoiseProfile()
            profile.parse(tlv_data)
            frame_data['tlvs'].append(profile)
        elif tlv_type == data.MMWDEMO_OUTPUT_MSG_STATS:
            stats = data.Statistics()
            stats.parse(tlv_data)
            frame_data['tlvs'].append(stats)
        elif tlv_type == data.MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO:
            side_info = data.SideInfoForDetectedPoints()
            side_info.parse(tlv_data)
            frame_data['tlvs'].append(side_info)
        elif tlv_type == data.MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS:
            temperature = data.TemperatureStatistics()
            temperature.parse(tlv_data)
            frame_data['tlvs'].append(temperature)
        elif tlv_type == data.MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS:
            point_cloud = data.SphericalPoints()
            point_cloud.parse(tlv_data)
            frame_data['tlvs'].append(point_cloud)
        elif tlv_type == data.MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST:
            target_list = data.TargetList()
            target_list.parse(tlv_data)
            frame_data['tlvs'].append(target_list)
        elif tlv_type == data.MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX:
            target_index = data.TargetIndex()
            target_index.parse(tlv_data)
            frame_data['tlvs'].append(target_index)
        else:
            tlv = data.TLV()
            tlv.parse(tlv_data)
            frame_data['tlvs'].append(tlv)
    
        tlv_data = tlv_data[tlv_length + 8:]
    return frame_data


def parse_file(path: str) -> List[Dict]:
    content = None
    with open(path, mode = "rb") as file:
        content = file.read()
    
    frames = list()
    while len(content) >= 8:
        frame_start = content.find(data.UART_MAGIC_WORD)
        if frame_start == -1:
            raise RuntimeError("No magic word found. Discard all data.")
        
        content = content[frame_start:]
        if len(content) < 40:
            raise RuntimeError("Not enough data for frame header. Discard all data.")
        
        packet_length = int.from_bytes(content[12:16], byteorder = 'little', signed = False)
        if len(content) < packet_length:
            raise RuntimeError("Not enough data for packet. Discard all data.")
        
        frame = parse_frame(content)
        frames.append(frame)
        content = content[packet_length:]
    
    return frames


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, data.PointCloud):
            return obj.__dict__
        elif isinstance(obj, data.RangeProfile):
            return obj.__dict__
        elif isinstance(obj, data.NoiseProfile):
            return obj.__dict__
        elif isinstance(obj, data.Statistics):
            return obj.__dict__
        elif isinstance(obj, data.SideInfoForDetectedPoints):
            return obj.__dict__
        elif isinstance(obj, data.TemperatureStatistics):
            return obj.__dict__
        elif isinstance(obj, data.SphericalPoints):
            return obj.__dict__
        elif isinstance(obj, data.TargetInfo):
            return obj.__dict__
        elif isinstance(obj, data.TargetList):
            return obj.__dict__
        elif isinstance(obj, data.TargetIndex):
            return obj.__dict__
        elif isinstance(obj, data.TLV):
            return obj.__dict__
        elif isinstance(obj, np.float32):
            return float(obj)
        else:
            return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path>")
        exit(1)
    
    files = sys.argv[1:]
    # files = ["dataset/01_15_2024_09_23_12"]
    for path in files:
        frames = parse_file(path)
        with open(f"{path}_dump.json", mode = "w") as file:
            json.dump(frames, file, cls = NumpyEncoder)
