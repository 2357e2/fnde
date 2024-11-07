# fnde
Fourier Neural Differential Equation model for learning quantum field theories.


## Installation
Clone repository using
```git clone https://github.com/2357e2/fnde```

To install, enter the download location in terminal and enter
```pip install fnde -e .```


## Basic use
To train a network navigate, to ```fnde/fnde/training``` and enter
```python main.py```

Parameters can be changed in the file ```fnde/fnde/training/parameters.py```

Important parameters include:
  - ```model_name_manual```: Select which model to run.
  - ```loss_func```: select which loss function to use.
  - ```non_linear```: select which activation function to use.
  - ```Hamiltonian```: select which theory to learn.
  - ```momenta_range```: number of momentum momenta discretizations used in the scattering matrix.
  - ```learning_rate```: learning rate.
  - ```epochs```: number of training epochs.
  - ```rewrite```: if True, the synthetic training data will be rewritten for the given parameters, if False, the previous synthetic training will be used.
