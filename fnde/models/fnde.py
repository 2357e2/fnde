import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from warnings import warn
from typing import List

from fnde.utils import log, FourierOperator, Projection, Lifting, contract_matrix_complex

class FNDE(nn.Module):
    """Fourier Neural Differential Equation model."""
    def __init__(
            self, batch_size: int, momenta: List[float], samp_ts: torch.Tensor, dim: int, rel_tol: float = 1e-7, abs_tol: float = 1e-7,
            input_size: int = 1, output_size: int = 1, channels: int = 10, 
            n_layers: int = 3, max_mode: int = 10, integrator: str = 'rk4', rank: float = 0.5, 
            norm: str = 'backward', projection_size: int = None, non_linear = None,
            in_bias: bool = True, out_bias: bool = False, normalise = None, integrate: bool = True
        ) -> None:
        super(FNDE, self).__init__()
        self.batch_size=batch_size
        self.momenta = momenta
        self.n_layers = n_layers
        self.max_mode = max_mode
        self.dim = dim
        self.input_size = input_size
        self.output_size = output_size
        self.channels = channels
        self.momenta = momenta
        self.integrator = integrator
        if not projection_size:
            projection_size = channels
        self.fourier_layers = nn.ModuleList([FourierLayer(batch_size=self.batch_size, channels=channels, max_mode=max_mode, dim=dim, n_layers=n_layers, norm=norm, rank=rank, in_bias = in_bias, out_bias = out_bias) for _ in range(n_layers)])
        if channels > 1:
            self.lifting = Lifting(input_size, channels)
        else:
            self.lifting = None
        if channels != output_size:    
            self.projection = Projection(channels, projection_size, output_size)
        else:
            self.projection = None
        self.non_linear = non_linear
        if integrate:
            self.integrate = ODESolver
        else:
            self.integrate = None
        if input_size != 1:
            self.input_bias = nn.ModuleList([nn.ModuleList([nn.Linear(input_size, 1) for i in range(dim)]) for j in range(dim)])
        else:
            self.input_bias = None
        self.samp_ts = samp_ts
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.normalise = normalise

    def forward(self, z):
        z = z.float()
        if self.input_bias:
            v = torch.zeros(self.batch_size, 1, self.dim, self.dim)
            for i in range(self.batch_size):
                v[i] = torch.tensor([[self.input_bias[j][k](z[i][0][j][k]) for j in range(self.dim)] for k in range(self.dim)])
        else:
            v = z

        if self.lifting:
            v = self.lifting(v)

        assert len(v.size()) == 4
        assert v.size()[1] == self.channels, v.size()
        out = v
        for fl in self.fourier_layers:
            if self.integrate:
                integrator = self.integrate(fl, self.samp_ts, self.rel_tol, self.abs_tol, integrator=self.integrator)
                out = integrator(out)[-1]
                # Take final time value
            else:
                out = fl(0, out)
            if self.non_linear is not None:
                out = self.non_linear(out)
        
        if self.projection:
            out = self.projection(out)

        if self.normalise:
            out = self.normalise(out, self.momenta)
        return out

class ODESolver(nn.Module):
    """
    The neural differential equation ODE solver class.
    """
    def __init__(self, odefunc, integration_times: torch.Tensor, rel_tol=1e-7, abs_tol=1e-7, integrator: str = 'rk4', log = False):
        super(ODESolver, self).__init__()
        self.odefunc = odefunc
        self.integration_times = integration_times
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.integrator = integrator
        self.log = log

    def forward(self, z):
        if self.log:
            z = torch.exp(z)
        try:
            out = odeint(self.odefunc, z, self.integration_times, rtol=self.rel_tol, atol=self.abs_tol, method=self.integrator)
        except AssertionError:
            warn(f"Underflow in odeint. Skip this integration.")
            out = z
        if self.log:
            z = log(z)
        return out
    
class FourierLayer(nn.Module):
    """
    Fourier Layer class, the largest class inside the integrator.
    """
    def __init__(self, batch_size, channels, max_mode, dim, n_layers, norm, rank = 0.5, mlp = None, in_bias = True, out_bias = False):
        super(FourierLayer, self).__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.max_mode = max_mode
        self.fourier_operator = FourierOperator(batch_size=batch_size, channels=channels, max_mode=max_mode, n_layers=n_layers, dim=dim, norm=norm, rank=rank, in_bias=in_bias, out_bias=out_bias, contraction=contract_matrix_complex)
        self.mlp = mlp

    def forward(self, t, z):
        out = self.fourier_operator(z)
        if self.mlp:
            lin = torch.transpose(self.mlp(torch.transpose(z, 0, -1)), 0, -1)
            out += lin
        return out
