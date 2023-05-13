import torch.nn as nn
import torch
from torchdiffeq import odeint

class NODE(nn.Module):
    """
    Neural Ordinary Differential Equation model.
    """
    def __init__(self, input_size: int, hidden_size: int, batch_size: int, output_size: int, n_layers: int = 3, dim = 10, non_linear = nn.Softplus()) -> None:
        super(NODE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.non_linear = non_linear
        self.dim = dim
        for i in range(n_layers):
            in_dim = hidden_size
            out_dim = hidden_size
            if i == 0:
                in_dim = input_size
            if i == (n_layers - 1):
                out_dim = output_size
            setattr(self, f"lin{i}", nn.Linear(in_dim, out_dim, dtype=float))

    def forward(self, t, z) -> torch.Tensor:
        out = NODE.data_in(z)
        for i in range(self.n_layers):
            out = getattr(self, f"lin{i}")(out)
            if i < self.n_layers - 1 and self.non_linear is not None:
                out = self.non_linear(out)
        out = NODE.data_out(out, self.batch_size, self.dim)
        return out
    
    @staticmethod
    def data_in(matrix: torch.Tensor) -> torch.Tensor:
        """Transforms rank 2 torch tensor into a rank 1 tensor, and returns with input_size"""
        batch_size, channels, dim, _ = matrix.size()
        vector = matrix.reshape(1, batch_size * dim**2)
        return vector
    
    @staticmethod
    def data_out(vector: torch.Tensor, batch_size: int, dim: int) -> torch.Tensor:
        """Transforms rank 1 torch tensor into a rank 2 tensor."""
        return vector.reshape((batch_size, 1, dim, dim))

    
class ODEIntegrator(nn.Module):
    """
    The neural differential equation ODE solver class.
    """
    def __init__(self, odefunction, integration_times, rel_tol, abs_tol, integrator, momenta, dim, batch_size, normalization = None):
        super(ODEIntegrator, self).__init__()
        self.odefunction = odefunction
        self.integration_times = integration_times
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.integrator = integrator
        self.momenta = momenta
        self.dim = dim
        self.batch_size = batch_size
        self.normalization = normalization
        self.input_bias = nn.ModuleList([nn.ModuleList([nn.Linear(2, 1) for i in range(dim)]) for j in range(dim)])
    
    def forward(self, z):
        z = z.float()
        if len(z.size()) > 4:
            v = torch.zeros((self.batch_size, 1, self.dim, self.dim), dtype=float)
            for i in range(self.batch_size):
                v[i] = torch.tensor([[self.input_bias[j][k](z[i][0][j][k]) for j in range(self.dim)] for k in range(self.dim)])
        else:
            v = z
        out = odeint(self.odefunction, v, self.integration_times, rtol=self.rel_tol, atol = self.abs_tol, method=self.integrator)[-1]
        if self.normalization:
            out = self.normalization(out, self.momenta)
        return out
        
