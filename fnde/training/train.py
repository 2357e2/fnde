import time
import numpy as np
import torch
from typing import Tuple

def train(
        s0: torch.Tensor, s: torch.Tensor, model, momenta, device, optimizer, epochs: int,
        loss_func, dim: int, batch_size: int, print_data_per_epoch: bool = False, scheduler = 
        None, validation_test = False, valid_initial = None, valid_actual = None
    ) -> Tuple:
    """
    Contains the model training loop. Iterates over each epoch, but not each batch element as the the 
    model has dimension [batch_size, in_channels, dim, dim].

    Parameters:
        - s0: The initial time true value tensor.
        - s: The final time true value tensor.
        - model: The neural network model, as instantiated in main().
        - device: torch.device.
        - optimizer: The model's ADAM optimizer, as instantiated in main().
        - loss_func: The model's loss function.
        - print_data_per_epoch (optional): Parameter to determine if the output should be printed
            per epoch (used primarily for debugging).
        - scheduler (optional): The learning rate scheduler used in the training loop. If given,
            the learning rate begins at the value defined in parameters.py If None, uses the constant
            learning rate defined in parameters.py.
    
    Return:
        - epoch array.
        - nfe array.
        - loss array.
        - time array.
        - validation loss array.
        - final predicted value of s (detached).
        - trained model.
    """
    epoch_arr = np.empty(epochs)
    loss_arr = np.empty(epochs)
    nfe_arr = np.empty(epochs)
    time_arr = np.empty(epochs)
    valid_loss_arr = np.empty(int(epochs / 5))

    s0 = torch.reshape(
        torch.tensor([[[[momenta[i]/momenta[-1], momenta[j]/momenta[-1]]] for i in range(dim) for j in range(dim) ],] * batch_size ), 
        (batch_size, 1, dim, dim, 2))

    for epoch in range(1, epochs + 1):
        model.nfe = 0
        epoch_start_time = time.time()
        
        optimizer.zero_grad()
        
        pred_s = model(s0).to(device)

        assert pred_s.size() == s.size(), f"train(): the dimension of pred_s does not match the dimension of s: pred_s.size(): {pred_s.size()}; s.size(): {s.size()}."

        loss = loss_func(pred_s, s)
        assert not torch.isnan(loss), "Training loop: received Loss not-a-number."
        loss.backward()
        loss_arr[epoch - 1] = loss

        optimizer.step()
        if scheduler:
            scheduler.step()
        
        epoch_end_time = time.time()

        if epoch % 5 == 0 and validation_test:
            valid_pred = model(valid_initial).to(device)
            valid_loss = loss_func(valid_pred, valid_actual)
            print(f"Validation loss: {valid_loss:.6}")
            valid_loss_arr[int((epoch-1)/5)] = valid_loss

        epoch_arr[epoch - 1] = epoch
        nfe_arr[epoch - 1] = model.nfe
        time_arr[epoch - 1] = epoch_end_time - epoch_start_time

        if print_data_per_epoch:
            print(f"s: {s}\n pred_s: {pred_s}")

        print(f"epoch: {epoch}, loss: {loss:.6}")

    loss = loss.detach().numpy()
    print(f"Final training loss: {loss:.6}")
    print(f"NFE: {model.nfe}")

    return epoch_arr, nfe_arr, loss_arr, time_arr, valid_loss_arr, pred_s.detach().numpy(), model