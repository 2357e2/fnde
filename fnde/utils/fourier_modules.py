from torch import nn
import torch
from tltorch.factorized_tensors.core import FactorizedTensor

class Lifting(nn.Module):
    """Lifts input to higher dimensional representation with convolutional layer."""
    def __init__(self, in_channels, out_channels):
        super(Lifting, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, z):
        assert type(z) is torch.Tensor, f"z passed to Lifting is not of type torch.Tensor, but type {type(z)}"
        return self.fc(z)
    
class Projection(nn.Module):
    """Projects input back onto the dimensional space of the output with convolutional layers."""
    def __init__(self, in_channels, hidden_channels, out_channels, non_linear = nn.Softplus()):
        super(Projection, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.non_linear = non_linear

    def forward(self, z):
        z = self.fc1(z)
        if self.non_linear:
            z = self.non_linear(z)
        z = self.fc2(z)
        return z    

class FourierOperator(nn.Module):
    """
    Transforms input into reciprocal space, filters, adds bias, and performs inverse transform.
    """
    def __init__(self, batch_size: int, channels: int, max_mode: int, n_layers: int, norm: str, dim: int, rank: float, contraction, factorization: str = 'cp', preactivation = None, in_bias: bool = False, out_bias: bool = False, fft: bool = True) -> None:
        super(FourierOperator, self).__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.max_mode = max_mode
        self.norm = norm
        self.rank = rank
        self.contraction = contraction
        self.half_n_modes = [max_mode//2, max_mode//2]
        self.n_layers = n_layers
        shape = (batch_size, channels, *self.half_n_modes)
        with torch.no_grad():
            self.weight = torch.empty(shape, dtype=torch.complex64, requires_grad=True)
            self.weight = self.weight.normal_(0, 1/dim**2)
        self.weight = nn.ModuleList([
            FactorizedTensor.from_tensor(
                self.weight,
                rank=rank, 
                factorization=factorization
            ) for _ in range(2)
        ])
        # Creating a FactorizedTensor disregards the imaginary components, but this is fine as interaction Hamiltonian is real-valued.
        self.preactivation = preactivation
        if in_bias:
            in_bias = torch.empty(*((batch_size, channels, dim, dim//2 + 1)), dtype=torch.cfloat)
            self.in_bias = nn.Parameter(in_bias.normal_(0, 1/dim**2))
        else:
            self.in_bias = None
        if out_bias:
            out_bias = torch.empty(*((batch_size, channels, dim, dim)), dtype=torch.float)
            self.out_bias = nn.Parameter(out_bias.normal_(0, 1/dim**2))
        else:
            self.out_bias = None
        self.fft = fft

    def forward(self, z):
        batch_size, channels, dim, dim = z.shape

        assert self.batch_size == batch_size, f"FourierOperator forward: self.batch_size == {self.batch_size} is not equal to tensor input batch_size == {batch_size}. Size: {z.shape}"
        assert self.channels == channels, f"FourierOperator forward: self.channels == {self.channels} is not equal to tensor input channels == {channels}. Size: {z.shape}"
        
        if self.preactivation:
            z = self.preactivation(z)

        out = z
        if self.fft:
            z_transform = torch.fft.rfft2(out.float(), dim=(-2,-1))
            z_transform.requires_grad_()
        
            # z_transform has shape [batch_size, channels, dim, dim//2 + 1]
            out_transform = torch.zeros([batch_size, channels, dim, dim//2 + 1], dtype=z.dtype, device=z.device)

            out_transform[:, :, :self.half_n_modes[0], :self.half_n_modes[1]] = self.contraction(
                z_transform[:, :, :self.half_n_modes[0], :self.half_n_modes[1]], 
                self.weight[0]
            )
            out_transform[:, :, -self.half_n_modes[0]:, :self.half_n_modes[1]] = self.contraction(
                z_transform[:, :, -self.half_n_modes[0]:, :self.half_n_modes[1]],
                self.weight[1]
            )
            if self.in_bias is not None:
                out_transform = out_transform + self.in_bias

            out = torch.fft.irfft2(out_transform, dim=(-2, -1), norm = self.norm)

        if self.out_bias is not None:
            out += self.out_bias
        return out