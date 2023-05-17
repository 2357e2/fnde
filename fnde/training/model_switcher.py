import torch

from fnde.models import *
from parameters import *
from fnde.utils import ParameterError

def switch_model(model_name, device, momenta: List[float], samp_ts: torch.Tensor):
    """
    Instantiates the selected model.

    Parameters:
        - model_name: the model to be instantiated.
        - device: torch.device.
        - momenta: the discretized momenta of the model.
        - samp_ts: the sampled times.

    Returns:
        - the instantiated model.
    """
    dim = len(momenta)
    if model_name == 'FNDE':
        model = FNDE(batch_size = batch_size, momenta=momenta, samp_ts = samp_ts, dim = dim, rel_tol = rel_tol, abs_tol = abs_tol, input_size=input_size, output_size=output_size, channels=lifting_channels, n_layers=fourier_layers, max_mode=max_mode, integrator = integrator, non_linear=non_linear, in_bias = False, out_bias = True, normalise = normalization, integrate=True).to(device)
    elif model_name == 'FNDE_mod':
        model = FNDE(batch_size = batch_size, momenta=momenta, samp_ts = samp_ts, dim = dim, rel_tol = rel_tol, abs_tol = abs_tol, input_size=input_size, output_size=output_size, channels=lifting_channels, n_layers=fourier_layers, max_mode=max_mode, integrator = integrator, non_linear=None, in_bias = True, out_bias = False, normalise = normalization, integrate=True).to(device)
    elif model_name == 'FNO':
        model = FNO(batch_size = batch_size, momenta=momenta, dim = dim, input_size=input_size, output_size=output_size, channels=lifting_channels, n_layers=fourier_layers, max_mode=max_mode, non_linear=non_linear, in_bias = False, out_bias = True, normalise = normalization).to(device)
    elif model_name == 'NODE':
        model = ODEIntegrator(
            NODE(
                input_size = batch_size*dim**2, output_size = batch_size*dim**2, batch_size = batch_size, hidden_size = hidden_size, n_layers = hidden_layers, dim=dim
            ), samp_ts, rel_tol, abs_tol, integrator=integrator, momenta = momenta, dim=dim, batch_size=batch_size, normalization=normalization
        ).to(device)    
    else:
        raise ParameterError('model_name', model_name)

    return model
