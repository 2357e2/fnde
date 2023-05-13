import torch
from data_handling import generate_s_matrix, Phi4_1_loop, Scalar_QED_tree, Scalar_Yukawa_tree
from typing import List, Tuple
from utils import conserve_momenta, RelativeLoss


# Save paths
save_path: str = 'C:/Users/fnde/fnde/data/'
inter_test_save_path: str = 'C:/Users/fnde/fnde/inter_data/'
extra_test_save_path: str = 'C:/Users/fnde/fnde/extra_data/'
validation_test_save_path: str = 'C:/Users/fnde/fnde/validation_data/'


# model parameters
model_names: List[str] = ['FNDE', 'FNDE_mod', 'FNO', 'NODE']
model_name_manual: str = model_names[0]

loss_funcs: List = [torch.nn.MSELoss(), torch.nn.L1Loss(), RelativeLoss()]
loss_func = loss_funcs[2]

non_linears: List = [torch.nn.Softplus(), torch.nn.ReLU()]
non_linear = non_linears[0]

Hamiltonians: List[torch.nn.Module] = [Phi4_1_loop, Scalar_QED_tree, Scalar_Yukawa_tree]
Hamiltonian = Hamiltonians[0]

generate_data = generate_s_matrix
normalization = conserve_momenta
integrator = 'rk4'
momenta_range: int = 10
momenta_precision: float = 1
mass: List[float] = [0.5, 1, 2, 5]
coupling: List[float] = [0.5, 1, 2, 5]
t_steps: int = 10
t0: float = 0
tN: float = 10
abs_tol: float = 1e-5
rel_tol: float = 1e-5
extrapolation_range: Tuple[float, float] = (0,2)
noise_std = 0.2


# hyperparameters
input_size = 2
output_size = 1
lifting_channels: int = 1
in_channels: int = 1
out_channels: int = 1
hidden_size: int = 100
learning_rate: float = 0.04 # initial learning rate if using a scheduler
epochs: int = 100
batch_size: int = len(mass) * len(coupling)
fourier_layers: int = 1
hidden_layers: int = 3
max_mode: int = 8
use_scheduler: bool = True
scheduler_gamma: float = 0.5
scheduler_steps: List[int] = [100,250]


# Regenerate data/test
rewrite: bool = True
inter_test: bool = False
extra_test: bool = False
print_data: bool = True
print_data_per_epoch: bool = False
validation_test = True