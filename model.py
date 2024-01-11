import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union, BinaryIO
from os import PathLike


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
        num_frames: int         = 10,
        num_points: int         = 64,
        num_features: int       = 4
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
        epsilon          = torch.empty(size = (batch_size, num_frames, latent_dimension)).normal_(mean = 0, std = 1.0)
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

    def load(self, state_dict: Union[str, PathLike, BinaryIO]) -> None:
        self.load_state_dict(torch.load(state_dict))
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
