import time
import serial
import serial.tools.list_ports
import random
import numpy as np
import torch
from typing import List, Tuple
from model import HVRAE
from data import TLV, PointCloud, UART_MAGIC_WORD, MMWDEMO_OUTPUT_MSG_DETECTED_POINTS


# HVRAE config
FRAMES_PER_PATTERN = 10
POINTS_PER_FRAME   = 64
FEATURES_PER_POINT = 4

# mmWave radar info
TILT_ANGLE   = 0.0 # degree
RADAR_HEIGHT = 1.2 # meter


def robust_z_score(data: np.ndarray) -> np.ndarray:
    median_x = np.median(data[:, 0])
    median_y = np.median(data[:, 1])
    median_z = np.median(data[:, 2])

    median = np.array([median_x, median_y, median_z])

    mad_x = np.median(np.abs(data[:, 0] - median_x))
    mad_y = np.median(np.abs(data[:, 1] - median_y))
    mad_z = np.median(np.abs(data[:, 2] - median_z))

    mad_dist = np.linalg.norm(np.abs(np.array([mad_x, mad_y, mad_z])))
    # print("median: {}, mad: {}, mad_dist: {}".format(median, np.array([mad_x, mad_y, mad_z]), mad_dist))

    result = list()
    for i in range(data.shape[0]):
        pos = data[i, 0:3]
        dist = np.linalg.norm(np.abs(pos - median))
        if 0.675 * dist < 3.0 * mad_dist:
            result.append(np.array([pos[0], pos[1], pos[2], data[i, -1]]))

    return np.array(result)


def oversampling(pattern: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
    rotation_matrix = np.array(
        [[1.0, 0.0, 0.0],
        [0.0, np.cos(np.deg2rad(TILT_ANGLE)), np.sin(np.deg2rad(TILT_ANGLE))],
        [0.0, -np.sin(np.deg2rad(TILT_ANGLE)), np.cos(np.deg2rad(TILT_ANGLE))]])

    # Center of the first frame.
    center = np.mean(pattern[0], axis=0)
    center = center[0:3]
    center = np.matmul(rotation_matrix, center)
    center[2] += RADAR_HEIGHT

    # Preprocess frames.
    preprocessed_pattern = []
    for frame in pattern:
        preprocessed_frame = []
        for point in frame:
            new_point = np.matmul(rotation_matrix, point[0:3])
            new_point[2] += RADAR_HEIGHT
            delta = np.array([new_point[0] - center[0], new_point[1] - center[1], new_point[2], point[3]])
            preprocessed_frame.append(delta)

        # Oversampling
        frame_np = np.array(preprocessed_frame)
        # Check empty frame.
        N = POINTS_PER_FRAME
        M = frame_np.shape[0]
        assert M != 0, "Empty frame."

        # Rescale and padding.
        mean = np.mean(frame_np, axis=0)
        frame_np = np.sqrt(N / M) * frame_np + mean - np.sqrt(N / M) * mean
        oversampled = frame_np.tolist()
        if len(oversampled) < N:
            oversampled.extend([mean] * (N - M))
        else:
            oversampled = random.shuffle(oversampled)
            oversampled = oversampled[:N]

        preprocessed_pattern.append(oversampled)
    
    # Convert to numpy array.
    oversampled_pattern = np.array(preprocessed_pattern)
    assert oversampled_pattern.shape[-2] == POINTS_PER_FRAME, "ERROR: The new frame has wrong number of points."
    assert oversampled_pattern.shape[-1] == FEATURES_PER_POINT, "ERROR: The new frame has wrong number of features."
    return oversampled_pattern, center


def calculate_height_diff(pattern: List[np.ndarray]) -> float:
    max_diff = 0
    for i in range(len(pattern)):
        for j in range(i + 1, len(pattern)):
            diff = pattern[i] - pattern[j]
            if diff > max_diff:
                max_diff = diff
    
    return max_diff


class FileParser:
    def __init__(self) -> None:
        self.pattern           = list()
        self.heights           = list()
        self.empty_frame_count = 0
        self.model             = HVRAE()
        self.model.load("model/HVRAE.pth")

    def exec(self, dataset: bytes, output) -> None:
        while len(dataset) > 0:
            frame_start = dataset.find(UART_MAGIC_WORD)
            if frame_start < 0:
                break
            dataset = dataset[frame_start:]
            assert len(dataset) >= 40, "ERROR: broken dataset."
            assert dataset[0:8] == UART_MAGIC_WORD, "ERROR: invalid frame header."

            num_tlvs   = int.from_bytes(dataset[32:36], byteorder="little", signed=False)
            pointcloud = PointCloud()

            dataset = dataset[40:]
            for _ in range(num_tlvs):
                tlv_type = int.from_bytes(dataset[0:4], byteorder="little", signed=False)
                tlv_len  = int.from_bytes(dataset[4:8], byteorder="little", signed=False)
                if tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS:
                    pointcloud.parse(dataset)
                dataset = dataset[tlv_len + 8:]
            
            if len(pointcloud.points) > 0:
                self.infer(output, pointcloud)

    def infer(self, output, tlv: PointCloud) -> None:
        frame = tlv.points
        self.pattern.append(frame)
        if frame.shape[0] == 0:
            self.empty_frame_count += 1
            self.heights.append(0)
        else:
            self.heights.append(np.mean(frame[:, 2], axis=0))
        
        while len(self.pattern) > FRAMES_PER_PATTERN:
            frame  = self.pattern.pop(0)
            if frame.shape[0] == 0:
                self.empty_frame_count -= 1
            self.heights.pop(0)
        
        if len(self.pattern) == FRAMES_PER_PATTERN and self.empty_frame_count == 0:
            oversampled_pattern, center = oversampling(self.pattern)
            height_diff = calculate_height_diff(self.heights)
            with torch.no_grad():
                pred = self.model.predict(torch.from_numpy(oversampled_pattern).unsqueeze(0).to(dtype=torch.float32))
                print(f"Anomaly: {pred}, Center: {center}")
                print(f"{time.time()},{center[0]},{center[1]},{center[2]},{pred},{pred >= 0.05 and height_diff > 0.3}", file=output)


if __name__ == "__main__":
    # torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    parser = FileParser()
    with open("normal_output.csv", "w") as file:
        data = None
        with open("dataset/in_room_normal.bin", "rb") as dataset:
            data = dataset.read()
        parser.exec(data, file)

    with open("fall_output.csv", "w") as file:
        data = None
        with open("dataset/outside_fall.bin", "rb") as dataset:
            data = dataset.read()
        parser.exec(data, file)
