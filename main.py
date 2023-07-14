import time
import serial
import random
import numpy as np
import torch
import os
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Any, Union, BinaryIO

# Pytorch default settings.
__PYTORCH_DEVICE__ = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(__PYTORCH_DEVICE__)

MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1
MMWDEMO_OUTPUT_MSG_RANGE_PROFILE = 2
MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
MMWDEMO_OUTPUT_MSG_STATS = 6
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO = 7
MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP = 8
MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS = 9
MMWDEMO_OUTPUT_EXT_MSG_DETECTED_POINTS = 301
MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS = 1000
MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST = 1010
MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX = 1011
MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT = 1012
MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS = 1020
MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION = 1021
MMWDEMO_OUTPUT_MSG_OCCUPANCY_STATE_MACHINE = 1030
MMWDEMO_OUTPUT_MSG_GESTURE_FEATURES = 1050
MMWDEMO_OUTPUT_MSG_ANN_OP_PROB = 1051

class StatisticsTLV:
    def __init__(self) -> None:
        self.inter_frame_processing_time = 0
        self.transmit_output_time = 0
        self.inter_frame_processing_margin = 0
        self.inter_chirp_processing_margin = 0
        self.active_frame_cpu_load = 0
        self.inter_frame_cpu_load = 0

class TemperatureStatisticsTLV:
    def __init__(self) -> None:
        self.temp_report_valid = 0
        self.time = 0
        self.tmp_rx0_sens = 0
        self.tmp_rx1_sens = 0
        self.tmp_rx2_sens = 0
        self.tmp_rx3_sens = 0
        self.tmp_tx0_sens = 0
        self.tmp_tx1_sens = 0
        self.tmp_tx2_sens = 0
        self.tmp_pm_sens = 0
        self.tmp_dig0_sens = 0
        self.tmp_dig1_sens = 0

class FrameParser:
    def __init__(self, port: str, baudrate: int) -> None:
        self.UART_MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self._serial = serial.Serial(port, baudrate)
        self._buffer = bytearray()

    def on_frame(self, frame_id: int, num_detected_obj: int, tlv: List[Tuple[int, Any]]) -> None:
        pass

    def should_stop(self) -> bool:
        return False

    def parse_frame(self) -> None:
        assert(len(self._buffer) >= 40)
        assert(self._buffer[0:8] == self.UART_MAGIC_WORD)

        frame_id = int.from_bytes(self._buffer[20:24], byteorder='little', signed=False)
        time = int.from_bytes(self._buffer[24:28], byteorder='little', signed=False)
        num_detected_obj = int.from_bytes(self._buffer[28:32], byteorder='little', signed=False)
        num_tlvs = int.from_bytes(self._buffer[32:36], byteorder='little', signed=False)

        # print(f"Frame ID: {frame_id}, Time: {time}, #Detected Obj: {num_detected_obj}, #TLVs: {num_tlvs}")

        tlv_data = self._buffer[40:]
        tlv_list = list()

        for _ in range(num_tlvs):
            tlv_type = int.from_bytes(tlv_data[0:4], byteorder='little', signed=False)
            tlv_length = int.from_bytes(tlv_data[4:8], byteorder='little', signed=False)

            # print(f"TLV Type: {tlv_type}, TLV Length: {tlv_length}")

            if tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS:
                assert(tlv_length % 16 == 0)
                assert(num_detected_obj == tlv_length // 16)
                detected_points = np.frombuffer(tlv_data[8:tlv_length + 8], dtype=np.float32).reshape((num_detected_obj, 4))
                tlv_list.append((MMWDEMO_OUTPUT_MSG_DETECTED_POINTS, detected_points))
            elif tlv_type == MMWDEMO_OUTPUT_MSG_STATS:
                assert(tlv_length == 24)
                stats = StatisticsTLV()
                stats.inter_frame_processing_time = int.from_bytes(tlv_data[8:12], byteorder='little', signed=False)
                stats.transmit_output_time = int.from_bytes(tlv_data[12:16], byteorder='little', signed=False)
                stats.inter_frame_processing_margin = int.from_bytes(tlv_data[16:20], byteorder='little', signed=False)
                stats.inter_chirp_processing_margin = int.from_bytes(tlv_data[20:24], byteorder='little', signed=False)
                stats.active_frame_cpu_load = int.from_bytes(tlv_data[24:28], byteorder='little', signed=False)
                stats.inter_frame_cpu_load = int.from_bytes(tlv_data[28:32], byteorder='little', signed=False)
                tlv_list.append((MMWDEMO_OUTPUT_MSG_STATS, stats))
            elif tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO:
                assert(num_detected_obj == tlv_length // 4)
                side_info = np.frombuffer(tlv_data[8:tlv_length + 8], dtype=np.uint16).reshape((num_detected_obj, 2))
                float_side_info = side_info.astype(np.float32) * 0.1
                tlv_list.append((MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO, float_side_info))
            elif tlv_type == MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS:
                assert(tlv_length == 28)
                temperature = TemperatureStatisticsTLV()
                temperature.temp_report_valid = int.from_bytes(tlv_data[8:12], byteorder='little', signed=False)
                temperature.time = int.from_bytes(tlv_data[12:16], byteorder='little', signed=False)
                temperature.tmp_rx0_sens = int.from_bytes(tlv_data[16:18], byteorder='little', signed=False)
                temperature.tmp_rx1_sens = int.from_bytes(tlv_data[18:20], byteorder='little', signed=False)
                temperature.tmp_rx2_sens = int.from_bytes(tlv_data[20:22], byteorder='little', signed=False)
                temperature.tmp_rx3_sens = int.from_bytes(tlv_data[22:24], byteorder='little', signed=False)
                temperature.tmp_tx0_sens = int.from_bytes(tlv_data[24:26], byteorder='little', signed=False)
                temperature.tmp_tx1_sens = int.from_bytes(tlv_data[26:28], byteorder='little', signed=False)
                temperature.tmp_tx2_sens = int.from_bytes(tlv_data[28:30], byteorder='little', signed=False)
                temperature.tmp_pm_sens = int.from_bytes(tlv_data[30:32], byteorder='little', signed=False)
                temperature.tmp_dig0_sens = int.from_bytes(tlv_data[32:34], byteorder='little', signed=False)
                temperature.tmp_dig1_sens = int.from_bytes(tlv_data[34:36], byteorder='little', signed=False)
                tlv_list.append((MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS, temperature))
            elif tlv_type == MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS:
                assert(tlv_length == 16 * num_detected_obj)
                coords = np.frombuffer(tlv_data[8:tlv_length + 8], dtype=np.float32).reshape((num_detected_obj, 4))
                tlv_list.append((MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS, coords))

            tlv_data = tlv_data[tlv_length + 8:]

        self.on_frame(frame_id, num_detected_obj, tlv_list)

    def run(self) -> None:
        while not self.should_stop():
            data = self._serial.read_all()
            self._buffer.extend(data)

            if len(self._buffer) < 8:
                continue

            frame_start = self._buffer.find(self.UART_MAGIC_WORD)
            if frame_start == -1:
                self._buffer.clear()
                continue
        
            self._buffer = self._buffer[frame_start:]
            if len(self._buffer) < 40:
                continue

            packet_length = int.from_bytes(self._buffer[12:16], byteorder='little', signed=False)
            if len(self._buffer) < packet_length:
                continue
            
            # print(f"Packet Length: {packet_length}")
            self.parse_frame()
            self._buffer = self._buffer[packet_length:]


def send_config(serial: serial.Serial, config: str | List[str]) -> None:
    if isinstance(config, str):
        time.sleep(0.03)
        if config[-1] != '\n':
            config += '\n'

        serial.write(config.encode())
        ack = serial.readline()
        print(ack)
        ack = serial.readline()
        print(ack)
        return
    
    for i, line in enumerate(config):
        if line == '\n':
            config.remove(line)
        elif line[-1] != '\n':
            config[i] += '\n'
    
    for line in config:
        time.sleep(0.03)

        serial.write(line.encode())
        ack = serial.readline()
        print(ack)
        ack = serial.readline()
        print(ack)

    time.sleep(0.03)
    serial.reset_input_buffer()


# HVRAE config
FRAMES_PER_PATTERN = 10
POINTS_PER_FRAME = 64
FEATURES_PER_POINT = 4

# mmWave radar info
TILT_ANGLE = 0.0 # degree
RADAR_HEIGHT = 1.0 # meter

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

# model
class RepeatVector(nn.Module):
    def __init__(self, count: int, from_dim: int) -> None:
        super(RepeatVector, self).__init__()
        self.from_dim     = from_dim
        self.repeat_count = count
    
    def forward(self, x: Tensor) -> Tensor:
        new_shape = (*x.shape[0 : self.from_dim], 1, *x.shape[self.from_dim :])
        y         = x.view(new_shape)

        shape = []
        for _ in new_shape:
            shape.append(1)
        
        shape[self.from_dim] = self.repeat_count
        y = y.repeat(shape)
        return y


class Encoder(nn.Module):
    def __init__(self, in_shape: torch.Size, out_channels: int) -> None:
        super(Encoder, self).__init__()
        
        intermidiate_channels = 64
        # frame_count = in_shape[0]
        
        # Calculate input channels
        data_shape = in_shape[1 :]
        in_channels = 1
        for i in data_shape:
            in_channels *= i

        self.flatten = nn.Flatten(start_dim = 2) # Time distributed flatten. dimension 0 is batch_size, dimension 1 is timestamp
        self.fc      = nn.Linear(in_channels, intermidiate_channels)
        self.tanh    = nn.Tanh()

        self.z_mean    = nn.Linear(intermidiate_channels, out_channels)
        self.z_log_var = nn.Linear(intermidiate_channels, out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.flatten(x)
        y = self.fc(y)
        y = self.tanh(y)

        z_mean    = self.z_mean(y)
        z_log_var = self.z_log_var(y)
        
        return (z_mean, z_log_var)


class RepeatVector(nn.Module):
    def __init__(self, count: int, from_dim: int) -> None:
        super(RepeatVector, self).__init__()
        self.from_dim     = from_dim
        self.repeat_count = count
    
    def forward(self, x: Tensor) -> Tensor:
        new_shape = (*x.shape[0 : self.from_dim], 1, *x.shape[self.from_dim :])
        y         = x.view(new_shape)

        shape = []
        for _ in new_shape:
            shape.append(1)
        
        shape[self.from_dim] = self.repeat_count
        y = y.repeat(shape)
        return y


class Encoder(nn.Module):
    def __init__(self, in_shape: torch.Size, out_channels: int) -> None:
        super(Encoder, self).__init__()
        
        intermidiate_channels = 64
        # frame_count = in_shape[0]
        
        # Calculate input channels
        data_shape = in_shape[1 :]
        in_channels = 1
        for i in data_shape:
            in_channels *= i

        self.flatten = nn.Flatten(start_dim = 2) # Time distributed flatten. dimension 0 is batch_size, dimension 1 is timestamp
        self.fc      = nn.Linear(in_channels, intermidiate_channels)
        self.tanh    = nn.Tanh()

        self.z_mean    = nn.Linear(intermidiate_channels, out_channels)
        self.z_log_var = nn.Linear(intermidiate_channels, out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.flatten(x)
        y = self.fc(y)
        y = self.tanh(y)

        z_mean    = self.z_mean(y)
        z_log_var = self.z_log_var(y)
        
        return (z_mean, z_log_var)


class HVRAE(nn.Module):
    def __init__(
        self,
        num_frames: int   = 10,
        num_points: int   = 64,
        num_features: int = 4
    ) -> None:
        super(HVRAE, self).__init__()

        self.num_points   = num_points
        self.num_features = num_features

        intermidiate_channels = 64
        latent_dimension      = 16

        self.z_mean    = None
        self.z_log_var = None

        # VAE: q(z|X). Input: motion pattern. Output: mean and log(sigma^2) for q(z|X).
        self.q_zx = Encoder((num_frames, num_points, num_features), intermidiate_channels)

        # RNN AutoEncoder
        self.encode        = nn.RNN(intermidiate_channels, latent_dimension, nonlinearity = "tanh", num_layers = 3)
        self.decode_repeat = RepeatVector(num_frames, 1)
        self.decode_rnn    = nn.RNN(latent_dimension, latent_dimension, nonlinearity = "tanh", num_layers = 3)
        
        # VAE: p(X|z). Output: mean and log(sigma^2) for p(X|z).
        self.x_latent    = nn.Linear(latent_dimension, intermidiate_channels)
        self.tanh        = nn.Tanh()
        self.p_xz_mean   = nn.Linear(intermidiate_channels, num_features)
        self.p_xz_logvar = nn.Linear(intermidiate_channels, num_features)

        self.p_xz_repeat = RepeatVector(num_points, 2)

    def forward(self, x: Tensor) -> Tensor:
        # Cache z_mean and z_log_var so that we can use them in the loss function.
        self.z_mean, self.z_log_var = self.q_zx(x)
        # print("z_mean device: {}, z_log_var device: {}.".format(self.z_mean.device, self.z_log_var.device))

        # VAE: sampling z ~ q(z|X) using reparameterization trick. Output: samples of z.
        z = self.sample(self.z_mean, self.z_log_var)
        
        # RNN Autoencoder. Output: reconstructed z.
        encoder_feature, _ = self.encode(z)
        encoder_feature    = encoder_feature[:, -1, :]
        decoder_feature    = self.decode_repeat(encoder_feature)
        decoder_feature, _ = self.decode_rnn(decoder_feature)
        decoder_feature    = decoder_feature.flip(-2)

        # VAE: p(X|z). Output: mean and log(sigma^2) for p(X|z).
        x_latent    = self.x_latent(decoder_feature)
        x_latent    = self.tanh(x_latent)
        p_xz_mean   = self.p_xz_mean(x_latent)
        p_xz_logvar = self.p_xz_logvar(x_latent)

        # Reshape the output. Output: (n_frames, n_points, n_features*2).
        # In each frame, every point has a corresponding mean vector with length of n_features and a log(sigma^2) vector with length of n_features.
        xz = torch.cat([p_xz_mean, p_xz_logvar], dim = -1)
        xz = self.p_xz_repeat(xz)
        xz = xz.view(xz.size(0), -1, self.num_points, 2 * self.num_features)

        return xz

    def sample(self, z_mean: Tensor, z_log_var: Tensor) -> Tensor:
        batch_size       = z_mean.size(0)
        num_frames       = z_mean.size(1)
        latent_dimension = z_mean.size(2)
        epsilon          = torch.empty(size = (batch_size, num_frames, latent_dimension)).normal_(mean = 0, std = 1.0).to(device = __PYTORCH_DEVICE__)
        # print("sample z_mean device: {}, z_log_var device: {}, epsilon device: {}.".format(z_mean.device, z_log_var.device, epsilon.device))
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
    
    def train_loss(self, y: Tensor, pred: Tensor) -> Tensor:
        batch_size   = y.size(0)
        num_frames   = y.size(1)
        num_features = y.size(-1)

        z_mean    = self.z_mean
        z_log_var = self.z_log_var

        mean   = pred[:, :, :, : num_features]
        logvar = pred[:, :, :, num_features :]
        var    = torch.exp(logvar)

        y_reshape = y.reshape(batch_size, num_frames, -1)
        mean      = mean.reshape(batch_size, num_frames, -1)
        var       = var.reshape(batch_size, num_frames, -1)
        logvar    = logvar.reshape(batch_size, num_frames, -1)

        # E[log_pXz] ~= log_pXz
        log_p_xz = torch.square(y_reshape - mean) / var
        log_p_xz = torch.sum(0.5 * log_p_xz, dim = -1)

        # KL divergence between q(z|X) and p(z)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var), dim = -1)
        return torch.mean(log_p_xz + kl_loss)
    
    def load(self, file: Union[str, os.PathLike, BinaryIO]) -> None:
        self.load_state_dict(torch.load(file))
        self.to(device = __PYTORCH_DEVICE__)
        self.eval()
    
    def predict(self, data: Tensor) -> float:
        def loss_function(y_true: Tensor, y_pred: Tensor, z_mean: Tensor, z_log_var: Tensor) -> float:
            batch_size   = y_true.size(0)
            num_frames   = y_true.size(1)
            num_features = y_true.size(-1)

            mean   = y_pred[:, :, :, : num_features]
            logvar = y_pred[:, :, :, num_features :]
            var    = torch.exp(logvar)

            y_reshape = y_true.reshape(batch_size, num_frames, -1)
            mean      = mean.reshape(batch_size, num_frames, -1)
            var       = var.reshape(batch_size, num_frames, -1)
            logvar    = logvar.reshape(batch_size, num_frames, -1)

            # E[log_pXz] ~= log_pXz
            log_p_xz = torch.square(y_reshape - mean) / var
            log_p_xz = torch.sum(0.5 * log_p_xz, dim = -1)

            # KL divergence between q(z|X) and p(z)
            kl_loss = -0.5 * torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var), dim = -1)
            return torch.mean(log_p_xz + kl_loss)

        with torch.no_grad():
            pred              = self.forward(data)
            z_mean, z_log_var = self.q_zx(data)
            loss              = loss_function(data, pred, z_mean, z_log_var)
            return loss.item()

class HVRAEParser(FrameParser):
    def __init__(self) -> None:
        super().__init__("COM4", 921600)
        self._pattern = []
        self._empty_frame_count = 0
        self._model = HVRAE()
        self._model.load("model/HVRAE.pth")
        self._out_file = open("out.csv", "a")

    def on_frame(self, frame_id: int, num_detected_obj: int, tlv: List[Tuple[int, Any]]) -> None:
        frame = robust_z_score(tlv[0][1]) if len(tlv[0][1]) > 0 else tlv[0][1]        
        self._pattern.append(frame)
        if frame.shape[0] == 0:
            self._empty_frame_count += 1

        while len(self._pattern) > FRAMES_PER_PATTERN:
            frame = self._pattern.pop(0)
            if frame.shape[0] == 0:
                self._empty_frame_count -= 1
        
        if len(self._pattern) == FRAMES_PER_PATTERN and self._empty_frame_count == 0:
            oversampled_pattern, center = oversampling(self._pattern)
            with torch.no_grad():
                pred = self._model.predict(torch.from_numpy(oversampled_pattern).unsqueeze(0).to(device = __PYTORCH_DEVICE__, dtype=torch.float32))
                print("Anomaly: {}, Center: {}".format(pred, center))

                self._out_file.write("{},{},{},{},{},{}\n".format(time.time(), pred, center[0], center[1], center[2], pred >= 0.1))
            # print(oversampled_pattern.shape)
            # print(center)


if __name__ == '__main__':
    cli = serial.Serial("COM3", 115200)

    with open('config2.cfg', 'r') as cfg:
        config = cfg.readlines()
        send_config(cli, config)

    parser = HVRAEParser()
    parser.run()
