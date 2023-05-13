import torch
import tensorly as tl
import numpy as np
from typing import Tuple
import torch.nn.functional as f
from typing import List

def contract_matrix_complex(z, weight):
    """
    Contracts 3rd and 4th indices of a complex z and complex weight by matrix multiplication.
    
    Parameters:
        - z: the tensor being pushed through the model.
        - weight: the weight tensor used to filter Fourier modes.

    Returns:
        - the matrix-like contraction of z and weight.
    """
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    size = z.size()
    assert size == weight.size(), f"Contract_matrix_complex: weight tensor dimensions {weight.size()} does not match input tensor dimensions {size}"
    contraction = torch.zeros((size), dtype=torch.complex64)
    for i in range(size[0]):
        for j in range(size[1]):
            contraction[i][j] = torch.matmul(weight[i][j], z[i][j])
    return contraction

def data_load(save_path, model_name, theory_name, device, file_names = ['samp_ts.npy', 's0.npy', 's.npy'], requires_grad = True, data_type = float):
    """
    Loads data from files as a gradient-requiring tensor.

    Parameters:
        - save_path: the base level file path.
        - model_name: the final level file path.
        - device: the device to which the data should be assigned.
        - file_names (optional): the list of final names from which the data should be 
            loaded. By default, it is samp_ts, s0, s.
        - requires_grad (optional): Boolean-valued parameter for if the data being loaded
            should be grad-requiring. Default true.

    Returns:
        - the data in tensorial form.
    """
    files = []
    for file_name in file_names:
        files.append(torch.tensor(np.load(f"{save_path}{model_name}./{theory_name}./{file_name}"), requires_grad=requires_grad, dtype=data_type).to(device)) # t_steps times in an array)
    return files

def conserve_momenta(tensor: torch.Tensor, momenta: List[float], absolute: bool = True):
    """
    A momentum conservation-sensitive per-row normalization. For a given tensor, every 'row' 
    (final index) will be normalised such that the total momentum of that row sums to 1.
    
    norm_row = row / sum_i (row[i] * momenta[i])
    Parameters:
        - tensor: the tensor to be normalised.
        - momenta: the list of discretized momenta with each element corresponding to the
            element of the tensor's row.
        - absolute (optional): determines if each product, row[i] * momenta[i] should have the
            absolute value taken. By default, False.
    Returns:
        - The row-normalised tensor.
    """
    assert type(tensor) is torch.Tensor, f"{type(tensor)}"
    dim = len(momenta)
    size = tensor.size()
    assert dim == size[-1]
    if len(size) > 1:
        out_tensor = torch.zeros(size)
        for i in range(size[0]):
            out_tensor[i] = conserve_momenta(tensor[i], momenta)
        return out_tensor
    else:
        if absolute:
            magnitude = sum(abs(tensor[i] * momenta[i]) for i in range(dim))
            return abs(tensor/magnitude)
        else:
            magnitude = sum(tensor[i] * momenta[i] for i in range(dim))
            return tensor/magnitude

def log(tensor: torch.Tensor):
    return torch.log(tensor + 1e-10) # Add a small number to avoid problems when element is zero.

def energy(momentum: float, mass: float) -> float:
    """Returns free particle energy."""
    return np.sqrt(momentum**2 + mass**2)

def inner_product(p1, p2, mass, add: bool):
    """Computes (p1 + p2)**2 or (p1 - p2)**2 four vector product."""
    if add:
        return (energy(p1, mass) + energy(p2, mass))**2 - (p1 + p2)**2
    else:
        return (energy(p1, mass) - energy(p2, mass))**2 - (p1 - p2)**2
