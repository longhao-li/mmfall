import torch
from torch import Tensor
from torch.nn import Module, Flatten, Linear, Tanh, RNN
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime


class MMFall(Module):
    def __init__(
        self,
        num_frames: int,
        num_points: int,
        channels: int,
        linear_hidden_size: int = 64,
        rnn_hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.num_points = num_points
        self.channels   = channels

        # cached data for training and predicting.
        self.z_mean_cache: None | Tensor   = None
        self.z_logvar_cache: None | Tensor = None
        
        self.encoder_flatten   = Flatten(start_dim=2)
        self.encoder_linear    = Linear(num_points * channels, linear_hidden_size)
        self.encoder_tanh      = Tanh()
        self.encoder_z_mean    = Linear(linear_hidden_size, linear_hidden_size)
        self.encoder_z_log_var = Linear(linear_hidden_size, linear_hidden_size)
        self.encoder_rnn       = RNN(linear_hidden_size, rnn_hidden_size, batch_first=True)

        self.decoder_rnn          = RNN(rnn_hidden_size, rnn_hidden_size, batch_first=True)
        self.decoder_x_latent     = Linear(rnn_hidden_size, linear_hidden_size)
        self.decoder_tanh         = Tanh()
        self.decoder_p_xz_mean    = Linear(linear_hidden_size, channels)
        self.decoder_p_xz_log_var = Linear(linear_hidden_size, channels)

    def forward(self, input: Tensor) -> Tensor:
        # Encode phase.
        x = self.encoder_flatten(input)
        x = self.encoder_linear(x)
        x = self.encoder_tanh(x)

        self.z_mean_cache    = self.encoder_z_mean(x)
        self.z_logvar_cache  = self.encoder_z_log_var(x)

        # Sampling z ~ q(z|X) using reparameterization trick. Output: samples of z.
        epsilon = torch.empty_like(self.z_mean_cache)
        epsilon.normal_(mean=0.0, std=1.0)
        z = self.z_mean_cache + torch.exp(0.5 * self.z_logvar_cache) * epsilon

        # RNN Autoencoder. Output: reconstructed z.
        x, _ = self.encoder_rnn(z)
        x    = x[:, -1, :]          # Get the last frame of features.
        x    = x.repeat((1, self.num_frames, 1))
        x, _ = self.decoder_rnn(x)
        x    = torch.flip(x, (1, ))   # Flip the time dimension.

        # VAE: p(X|z). Output: mean and log(sigma^2) for p(X|z).
        x            = self.decoder_x_latent(x)
        x            = self.decoder_tanh(x)
        p_xz_mean    = self.decoder_p_xz_mean(x)
        p_xz_log_var = self.decoder_p_xz_log_var(x)

        # Reshape the output. Output: (num_frames, num_points, num_features * 2).
        # In each frame, every point has a corresponding mean vector with length of n_features and a
        # log(sigma^2) vector with length of n_features.
        x = torch.cat((p_xz_mean, p_xz_log_var), dim=-1)
        x = x.unsqueeze(2)
        x = x.repeat((1, 1, self.num_points, 1))

        return x

    def loss(self, y: Tensor, pred: Tensor) -> Tensor:
        batch_size = y.size(0)

        z_mean    = self.z_mean_cache
        z_log_var = self.z_logvar_cache

        mean    = pred[:, :, :, :self.channels]
        log_var = pred[:, :, :, self.channels:]
        var     = torch.exp(log_var)

        y_reshape = y.reshape((batch_size, self.num_frames, -1))
        mean      = mean.reshape((batch_size, self.num_frames, -1))
        log_var   = log_var.reshape((batch_size, self.num_frames, -1))
        var       = var.reshape((batch_size, self.num_frames, -1))

        # E[log_pXz] ~= log_pXz
        log_p_xz = torch.square(y_reshape - mean) / var
        log_p_xz = torch.sum(0.5 * log_p_xz, dim=-1)

        # KL divergence between q(z|X) and p(z)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var), dim=-1)
        return torch.mean(log_p_xz + kl_loss)

    def predict(self, data: Tensor) -> float:
        with torch.no_grad():
            pred = self.forward(data)
            return self.loss(data, pred)

    def fit(self, train: DataLoader, test: DataLoader, epochs: int) -> None:
        timestamp  = datetime.now().strftime('%Y%m%d-%H%M%S')
        writer     = SummaryWriter(f"runs/{timestamp}")
        adam       = Adam(self.parameters(), lr=0.001)
        best_vloss = float('inf')

        for epoch in range(epochs):
            # Enable training mode.
            self.train(True)

            running_loss = 0.0
            last_loss = 0.0
            
            for i, data in enumerate(train):
                adam.zero_grad()
                pred = self.forward(data)
                loss = self.loss(data, pred)
                loss.backward()
                adam.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    last_loss = running_loss / 100
                    print(f"[{epoch + 1}, {i + 1}] loss: {last_loss}")
                    writer.add_scalar('training loss', last_loss, epoch * len(train) + i)
                    running_loss = 0.0

            # Test: to be written.
            self.eval()
            running_vloss = 0.0

            with torch.no_grad():
                for i, data in enumerate(test):
                    vloss = self.predict(data)
                    running_vloss += vloss
            
            avg_vloss = running_vloss / len(test)
            print(f"[{epoch + 1}] validation loss: {avg_vloss}")

            writer.add_scalar('validation loss', avg_vloss, epoch)
            writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.state_dict(), f'runs/{timestamp}/best.pth')
            
            torch.save(self.state_dict(), f'runs/{timestamp}/last.pth')

