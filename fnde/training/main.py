import torch
import torch.nn as nn
import numpy as np
import time

from fnde.utils import data_load
from parameters import *
from model_switcher import switch_model
from train import train


def main(model_name: str, repeat: str='0', exp: str='___') -> None:
    """
    The method responsible for initiating data generation, data loading, model instantiation,
    model training, data and model saving, and model testing.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"model: {model_name}\n")
    print(f"device: {device}\n\n")

    dim = int(momenta_range / momenta_precision)
    
    if rewrite:
        start_time = time.time()
        momenta = np.linspace(1, momenta_range, int(momenta_range/momenta_precision))
        samp_ts_ = torch.linspace(t0, tN, t_steps)
        generate_data(save_path=save_path, model_name=model_name, Theory=Hamiltonian, exp=exp, repeat=repeat, momenta=momenta, coupling=coupling, in_channels=1, samp_ts=samp_ts_, batch_size=batch_size, mass = mass, normalization=normalization)
        end_time = time.time()
        print(f"Data generated in {end_time-start_time:.1} seconds.")
    
    samp_ts, s0, s = data_load(save_path, model_name, f"{exp}{repeat}{Hamiltonian.__name__}", device)
    s0 = s0.detach()
    print('Data loaded')
    print(f"First matrix in training batch: {s[0]}")
    
    model = switch_model(model_name=model_name, device=device, momenta=momenta, samp_ts=samp_ts)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps,gamma=scheduler_gamma)
    else:
        scheduler = None

    if validation_test:
        valid_momenta = [(i + 0.5 + np.random.rand()) * momenta_precision for i in range(dim)]

        generate_s_matrix(save_path=validation_test_save_path, model_name=model_name, Theory=Hamiltonian, momenta=valid_momenta, coupling=coupling, in_channels=1, samp_ts=samp_ts_, batch_size=batch_size, mass=mass, normalization=normalization, exp = exp, repeat = repeat)

        _, valid_initial_, valid_actual = data_load(validation_test_save_path, model_name, f"{exp}{repeat}{Hamiltonian.__name__}", device)
        valid_initial = torch.reshape(
        torch.tensor([[[[valid_momenta[i]/momenta_range, valid_momenta[j]/momenta_range]] for i in range(dim) for j in range(dim) ],] * batch_size ), 
        (batch_size, 1, dim, dim, 2))
    else:
        valid_initial = None
        valid_actual = None

    start_time = time.time()
    epoch_arr, nfe_arr, loss_arr, time_arr, valid_loss_arr, pred_s, model = train(
        s0=s0, s=s, model=model, momenta=momenta, device=device, optimizer=optimizer, scheduler=scheduler,
        epochs=epochs, loss_func=loss_func, dim=dim, batch_size=batch_size, print_data_per_epoch=print_data_per_epoch, validation_test=validation_test, valid_initial = valid_initial, valid_actual=valid_actual
    )
    end_time = time.time()

    print(f"Training time: {end_time - start_time:.3} seconds")
    
    if print_data:
        print(f"final time step pred_s: {pred_s[-1]}")
        print(f"final time step s: {s[-1]}")

    np.save(f"{save_path}{model_name}/{exp}{repeat}{Hamiltonian.__name__}/epoch_arr", epoch_arr)
    np.save(f"{save_path}{model_name}/{exp}{repeat}{Hamiltonian.__name__}/nfe_arr", nfe_arr)
    np.save(f"{save_path}{model_name}/{exp}{repeat}{Hamiltonian.__name__}/loss_arr", loss_arr)
    np.save(f"{save_path}{model_name}/{exp}{repeat}{Hamiltonian.__name__}/time_arr", time_arr)
    np.save(f"{save_path}{model_name}/{exp}{repeat}{Hamiltonian.__name__}/s_pred", pred_s)
    if validation_test:
        np.save(f"{validation_test_save_path}{model_name}/{exp}{repeat}{Hamiltonian.__name__}/valid_loss_arr", valid_loss_arr)
    torch.save(model, f"{save_path}{model_name}/{exp}{repeat}{Hamiltonian.__name__}/{model_name}_{Hamiltonian.__name__}_model.pth")
    
    print('\nprocess sucessfully terminated.\n\n')


if __name__ == '__main__':
    main(model_name_manual)