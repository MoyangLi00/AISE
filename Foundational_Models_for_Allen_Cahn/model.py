import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]
        # hint: use torch.fft library torch.fft.rfft
        # use DFT to approximate the fourier transform

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNOBlock(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        # TODO: Initialize the FNO block components
        self.modes1 = modes
        self.width = width
        self.padding = 1  # pad the domain if input is non-periodic
        self.linear_p = nn.Linear(2, self.width)  # input channel is 2: (u0(x), x) --> GRID IS INCLUDED!

        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width * 3, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = torch.nn.Tanh()
        
    def fourier_layer(self, x, spectral_layer, conv_layer):
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, x, time_embedding, epsilon_embedding):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.linear_p(x)
        x = x.permute(0, 2, 1)

        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.fourier_layer(x, self.spect3, self.lin2)

        # x = x[..., :-self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)  # [batch, 128, 64]
        bs, len, width = x.shape
        ts_len = time_embedding.shape[1]
        x = x.reshape(bs, 1, len, width)
        x = x.repeat(1, ts_len, 1, 1)
        x = x.reshape(bs * ts_len, len, width)
        # [bs, ts_len, width] -> [bs * ts_len, 1, width]
        time_embedding = time_embedding.reshape(bs * ts_len, 1, width).repeat(1, len, 1)
        epsilon_embedding = epsilon_embedding.unsqueeze(1).repeat(1, ts_len, 1)
        epsilon_embedding = epsilon_embedding.reshape(bs * ts_len, 1, width).repeat(1, len, 1)

        x = torch.concat((x, time_embedding, epsilon_embedding), dim=-1)

        x = self.linear_layer(x, self.linear_q)
        x = self.output_layer(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(1, embedding_dim)
        position = torch.arange(0, 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        # Apply sine to even indices
        pe[0, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[0, 1::2] = torch.cos(position * div_term)
        
        # Register the positional encoding as a buffer (won't be updated during training)
        self.register_buffer('pe', pe)

    def forward(self, t):
        """
        Args:
            t: Tensor of shape (batch_size, 1) containing time values
        Returns:
            Tensor of shape (batch_size, embedding_dim) containing time embeddings
        """
        # Scale time values to [0, 1] range if they aren't already
        t_scaled = t.clone()
        if t.max() > 1 or t.min() < 0:
            t_scaled = (t - t.min()) / (t.max() - t.min())
        
        # Compute embeddings using scaled time values
        t_scaled = t_scaled.view(-1, 1)  # Ensure shape is (batch_size, 1)
        embeddings = torch.zeros(t_scaled.shape[0], self.embedding_dim, device=t.device)    # [128, 64]
        
        for pos in range(self.embedding_dim // 2):
            omega = 1.0 / (10000 ** (2 * pos / self.embedding_dim))
            embeddings[:, 2*pos] = torch.sin(t_scaled * omega).squeeze(-1)
            embeddings[:, 2*pos+1] = torch.cos(t_scaled * omega).squeeze(-1)

        embeddings = embeddings.reshape(t.shape[0], t.shape[1], self.embedding_dim)
            
        return embeddings

class AllenCahnFNO(nn.Module):
    def __init__(self, modes=16, width=64):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # TODO: Initialize model components
        # Consider:
        # - Epsilon embedding
        # - Time embedding
        # - Input/output layers
        # - FNO blocks
        self.epsilon_embedding = nn.Linear(1, width)
        self.time_embedding = TimeEmbedding(width)
        self.fno_blocks = FNOBlock(modes, width)
        self.get_grid = lambda shape, device: torch.linspace(-1, 1, shape[-1], device=device).view(1, 1, -1)

    def forward(self, x, eps, t):
        """
        Args:
            x: Initial condition (batch_size, x_size)
            eps: Epsilon values (batch_size, 1)
            t: Time points (batch_size, n_steps)
        Returns:
            Predictions at requested timepoints (batch_size, n_steps, x_size)
        """
        # TODO: Implement the full model forward pass
        # 1. Embed epsilon and time
        # print(x.shape)
        # print(self.epsilon_embedding(eps).shape)    # [32, 64]
        # print(self.time_embedding(t).shape)         # [32, 4, 64]
        bs, len = x.shape
        grid = self.get_grid(x.shape, x.device).repeat(x.shape[0], 1, 1)

        x = torch.cat((x.unsqueeze(1), grid), dim=1).permute(0, 2, 1)

        # 2. Process spatial information with FNO blocks
        time_embedding = self.time_embedding(t)
        epsilon_embedding = self.epsilon_embedding(eps)
        x = self.fno_blocks(x, time_embedding, epsilon_embedding)

        # 3. Generate predictions for each timestep
        x = x.reshape(bs, -1, len)
        return x
        

def get_loss_func():
    """
    TODO: Define custom loss function(s) for training
    Consider:
    - L2 loss on predictions
    - Physical constraints (energy, boundaries)
    - Gradient-based penalties
    """
    loss = nn.MSELoss()
    return loss

def get_optimizer(model):
    """
    TODO: Configure optimizer and learning rate schedule
    Consider:
    - Adam with appropriate learning rate
    - Learning rate schedule for curriculum
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return optimizer

def train_step(model, batch, optimizer, loss_func):
    """
    TODO: Implement single training step
    1. Forward pass
    2. Loss computation
    3. Backward pass
    4. Optimizer step
    Return loss value
    """
    output = model(batch['initial'], batch['epsilon'], batch['times'])
    loss = loss_func(output, batch['target'])
    loss.backward()
    optimizer.step()
    return loss.item()

def validation_step(model, batch, loss_func):
    """
    TODO: Implement single validation step
    Similar to train_step but without gradient updates
    Return loss value
    """
    output = model(batch['initial'], batch['epsilon'], batch['times'])
    loss = loss_func(output, batch['target'])
    return loss.item(), output

# Example usage:
if __name__ == "__main__":
    # Model initialization
    model = AllenCahnFNO(modes=16, width=64)
    
    # Sample data
    batch_size, x_size = 32, 128
    x = torch.randn(batch_size, x_size)
    eps = torch.randn(batch_size, 1)
    t = torch.linspace(0, 1, 4)[None].expand(batch_size, -1)
    
    # Forward pass
    output = model(x, eps, t)  # Should return (batch_size, n_steps, x_size)
