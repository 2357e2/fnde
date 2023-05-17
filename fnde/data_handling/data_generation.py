from torch import nn
import torch
import numpy as np
import os
from typing import Union, Tuple, List

from fnde.utils import inner_product
    
class Phi4_1_loop(nn.Module):
    """
    1-loop level Phi 4 Theory data generation class.
    """
    def __init__(
            self, ext_momenta: List[float], coupling: List[float], mass: List[float], 
            dim: int, output = np.ndarray
        ):
        super(Phi4_1_loop, self).__init__()
        self.mass = mass
        self.coupling = coupling
        self.dim = dim
        # Cross symmetry approx. for p, q
        self.s = lambda p, q: (p + q)**2
        self.output = output
        self.matrix_element = lambda p, q, s: (1 / (np.sqrt(2 * p * 2 * q)) 
        * (-coupling - coupling**2/(4 * np.pi)**2 * np.log(s) - 5 * coupling**3/(2 * (4 * np.pi) ** 4) * (np.log(s))**2)) # from peskin schr exercise 10.4 page 345

    def forward(self, momenta):
        """
        Returns the synthetically-generated scattering matrix for the given momentum input. 
        Output matrix will be the of dimension nxn where n is the dimension of momenta.
        """
        matrix = torch.tensor(np.identity(self.dim), dtype=float)
        for i in range(self.dim):
            for j in range(self.dim):
                matrix[i][j] += abs(self.matrix_element(momenta[i], momenta[j], self.s(momenta[i], momenta[j])))
        return matrix

class Scalar_QED_tree(nn.Module):
    """
    Tree-level Scalar QED data generation class.
    """
    def __init__(self, ext_momenta: List[float], coupling, mass, dim, output = np.ndarray):
        super(Scalar_QED_tree, self).__init__()
        self.ext_momenta = ext_momenta
        self.coupling = coupling
        self.mass = mass
        self.dim = dim
        self.output = output
        self.matrix_element = lambda p_in, p_out, k: 2 * coupling**2 * (1 + 1/(np.sqrt(mass**2 + p_in**2) * ext_momenta[k]) * p_in * p_out)
    
    def forward(self, momenta, k: int = 0):
        """
        Returns the synthetically-generated scattering matrix for the given momentum input. 
        Output matrix will be the of dimension nxn where n is the dimension of momenta.
        """
        matrix = np.identity(self.dim)
        if self.output is torch.Tensor:
            matrix = torch.tensor(matrix, dtype=float)
        for i in range(self.dim):
            for j in range(self.dim):
                matrix[i][j] += abs(self.matrix_element(momenta[i], momenta[j], k))
        return matrix

class Scalar_Yukawa_tree(nn.Module):
    """
    Tree-level Scalar Yukawa data generation class. 
    """
    def __init__(
            self, ext_momenta: List[float], coupling: List[float], mass: List[float], 
            dim: int, output = np.ndarray
        ):
        super(Scalar_Yukawa_tree, self).__init__()
        self.ext_momenta = ext_momenta
        self.coupling = coupling
        self.mass = mass
        self.dim = dim
        self.output = output
        self.s_channel = lambda p1, p2: 1 / (inner_product(p1, p2, mass, add=True) - mass**2)
        self.t_channel = lambda p1, p3: 1 / (inner_product(p1, p3, mass, add=False) - mass**2)
        self.u_channel = lambda p2, p3: 1 / (inner_product(p3, p2, mass, add=False) - mass**2)
    
    def forward(self, momenta: List[float], k: int = 0):
        """
        Returns the synthetically-generated scattering matrix for the given momentum input. 
        Output matrix will be the of dimension nxn where n is the dimension of momenta.
        """
        matrix = np.identity(self.dim)
        if self.output is torch.Tensor:
            matrix = torch.tensor(matrix, dtype=float)
        for i in range(self.dim):
            for j in range(self.dim):
                matrix[i][j] += abs(self.coupling**2 * (self.s_channel(momenta[i], self.ext_momenta[k]) + self.t_channel(momenta[i], momenta[j]) + self.u_channel(self.ext_momenta[k], momenta[j])))
        return matrix

def save_data(save_path: str, model_name: str, theory_name: str, file_name: Union[str, List[str]], data) -> None:
    """
    Saves data as a numpy file with the requested save path and file name.
    
    If file_name is a list, interprets structure of data to be
    [file_name_1, file_name_2, ...] <--> [datum_1, datum_2, ...]
    And saves each element of data with the corresponding file_name.

    Parameters:
        - save_path: the base save path.
        - model_name: the final folder level to be saved.
        - file_name: the name of the file to be created.
        - data: the data to be saved as a numpy file.
    """
    if type(file_name) is not str:
        for name, file in zip(file_name, data):
            assert type(name) is str
            save_data(save_path, model_name, theory_name, name, file)
    else:        
        try:
            assert type(file_name) is str
            np.save(f"{save_path}{model_name}./{theory_name}./{file_name}", data)
        except FileNotFoundError:
            try:
                os.mkdir(f"{save_path}{model_name}./{theory_name}")
                file = open(f"{save_path}{model_name}./{theory_name}./{file_name}", 'x')
                file.close()
                np.save(f"{save_path}{model_name}./{theory_name}./{file_name}", data)
            except FileNotFoundError:
                os.mkdir(f"{save_path}{model_name}")
                os.mkdir(f"{save_path}{model_name}./{theory_name}")
                file = open(f"{save_path}{model_name}./{theory_name}./{file_name}", 'x')
                file.close()
                np.save(f"{save_path}{model_name}./{theory_name}./{file_name}", data)

def generate_s_matrix(save_path: str, model_name: str, Theory, momenta: List[float], coupling: List[float], in_channels: int, samp_ts, batch_size: int, mass: List[float], exp: str = '', repeat: str = '', normalization = None, save: bool = True, noise_std: float = 0) -> Union[None, Tuple]:
    """
    Generates scattering matrix for a given Hamiltonian (Theory). Saves the S matrix along with the s0 initial values and sample times.
    Parameters:
        - save_path: path to save.
        - model_name: name of model.
        - Theory: class of the theory's Hamiltonian.
        - momenta: list of momenta.
        - coupling: list of coupling constants.
        - in_channels: second dimension of tensor output.
        - samp_ts: sample times.
        - batch_size: batch_size
        - mass: list of masses
        - normalization (optional): the normalization function applied to the s matrix.
        - save (optional): if the data should be saved (True) or returned (False).
    Return:
        - None.
        or
        - sample times.
        - true inital value of s.
        - true final value of s.
    """
    if type(coupling) is (float or int):
        coupling = [coupling]
    if type(mass) is (float or int):
        mass = [mass]
    coupling_quantity = len(coupling)
    mass_quantity = len(mass)
    dim = len(momenta)
    
    assert batch_size == coupling_quantity * mass_quantity, f"Model batch_size {batch_size} must be equal to the product of the number of masses, {mass_quantity}, and number of coupling constants, {coupling_quantity}, {coupling * mass_quantity}."

    assert in_channels <= dim, f"in_channels should be of size no greater than dim, {dim}, but received {in_channels}."

    s0 = np.array([[np.identity(dim),]*in_channels,]*batch_size)
    
    s = np.zeros((batch_size, in_channels, dim, dim))
    for i in range(batch_size):
        mass_index = i % mass_quantity
        coup_index = i // mass_quantity
        theory = Theory(
            momenta[:in_channels],
            coupling = coupling[coup_index],
            mass = mass[mass_index], 
            dim = dim,
            output=np.ndarray
        )
        if in_channels == 1:
            scat_mat = theory(momenta)
            s[i][0] = np.array(scat_mat)
            if noise_std > 1e-9:
                noise_mat = np.random.normal(0, noise_std, (dim, dim))
                s[i][0] += noise_mat
        else:
            for k in range(in_channels):
                scat_mat = theory(momenta, k)
                s[i][k] = np.array(scat_mat)
                if noise_std > 1e-9:
                    noise_mat = np.random.normal(0, noise_std, (dim, dim))
                    s[i][k] += noise_mat
    
    if normalization:
        s = normalization(torch.tensor(s), momenta)

    if save:
        save_data(save_path=save_path, model_name=model_name, theory_name=f"{exp}{repeat}{Theory.__name__}", file_name=["samp_ts", "s0", "s"], data=[samp_ts, s0, s])
    else:
        return torch.tensor(samp_ts, requires_grad=True), torch.tensor(s0, requires_grad=False), torch.tensor(s, requires_grad=True)
