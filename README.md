# ferminet_tum - Fermionic Neural Network
I wrote this library for my internship at TUM. It reproduces the results of [Pfau et al. (2020)](https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429 "Ab initio solution of the many-electron Schrödinger equation with deep neural networks"), which trains so-called Fermionic Neural Network to find the ground state wave functions of an atom or a molecule using neural network quantum states, see [Carleo and Troyer (2017)](https://dx.doi.org/10.1126/science.aag2302 "Solving the quantum many-body problem with artificial neural networks").

## Installation

ferminet_tum uses [JAX](https://github.com/google/jax) for the automatic differentiation, [Chex](https://github.com/deepmind/chex) for the initialization of the network parameters, [optax](https://github.com/deepmind/optax) for the optimization of the network parameters, [h5py](https://github.com/h5py/h5py) for storing the variational energy values, and [mendeleev](https://github.com/lmmentel/mendeleev) for the chemical elements used in the library.

To install it, you can run
```
git clone https://github.com/cevenkadir/ferminet_tum
cd ferminet_tum
python setup.py install --user
``` 

## Usage
Here is an example for a Lithium atom.


```python
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from ferminet_tum.atom import Atom
from ferminet_tum.network import FermiNet
from ferminet_tum.hamiltonian import Hamiltonian
from ferminet_tum.sampler import Sampler
from ferminet_tum.driver import NNQS
```


```python
li_atom = [Atom(id="Li", pos=jnp.array([0.0, 0.0, 0.0]))]
n_electrons = [2, 1]  # number of up- and down-spin electrons

n_1 = [8, 8]  # numbers of hidden units for the one-electron stream
n_2 = [4, 4]  # numbers of hidden units for the two-electron stream
n_k = 16  # number of many-electron determinants
L = len(n_1)  # number of layers

# initialize the FermiNet
fn = FermiNet(
    li_atom,
    n_electrons,
    L,
    n_k,
    n_1,
    n_2,
)
params = fn.init_params()  # initialize the parameters
```


```python
# initialize the Hamiltonian
hamiltonian = Hamiltonian(fn)
```


```python
n_step = 10  # number of steps in each sampling
step_std_per_dim = (
    0.02  # standard deviation of the Gaussian noise in each step for each dimension
)

# initalize the sampler
sampler = Sampler(fn, n_step, step_std_per_dim)
```


```python
n_samples = 4096  # number of samples

# initialize the driver
nnqs = NNQS(sampler, hamiltonian, fn, n_samples)
```


```python
n_iters = 7000  # number of iterations

# training the network
params, energies = nnqs.train(n_iters, params, optax.adam(learning_rate=1e-3))
```


      0%|          | 0/7000 [00:00<?, ?it/s]


    /opt/homebrew/Caskroom/miniforge/base/envs/jax-env/lib/python3.9/site-packages/jax/_src/ops/scatter.py:87: FutureWarning: scatter inputs have incompatible types: cannot safely cast value from dtype=int32 to dtype=bool. In future JAX releases this will result in an error.
      warnings.warn("scatter inputs have incompatible types: cannot safely cast "
    /opt/homebrew/Caskroom/miniforge/base/envs/jax-env/lib/python3.9/site-packages/jax/_src/ops/scatter.py:87: FutureWarning: scatter inputs have incompatible types: cannot safely cast value from dtype=int32 to dtype=bool. In future JAX releases this will result in an error.
      warnings.warn("scatter inputs have incompatible types: cannot safely cast "



```python
expected_energy = -7.47798  # in Hartree energy
```


```python
fig, ax = plt.subplots(dpi=150)

ax.plot(energies, label="FermiNet")
ax.set_xlabel("iteration")
ax.set_ylabel("<E>")
ax.axhline(expected_energy, color="r", linestyle="--", label="true energy")
ax.legend()
ax.patch.set_facecolor("white")
fig.patch.set_facecolor("white")
```


    
![png](README_files/README_10_0.png)
    


Here is an example for a LiH molecule.


```python
lih_mol = [
    Atom(id="Li", pos=jnp.array([0.0, 0.0, 3.015 / 2])),
    Atom(id="H", pos=jnp.array([0.0, 0.0, -3.015 / 2])),
]
n_electrons = [2, 2]  # number of up- and down-spin electrons

n_1 = [16, 16, 16]  # numbers of hidden units for the one-electron stream
n_2 = [8, 8, 8]  # numbers of hidden units for the two-electron stream
n_k = 16  # number of many-electron determinants
L = len(n_1)  # number of layers

# initialize the FermiNet
fn_mol = FermiNet(
    lih_mol,
    n_electrons,
    L,
    n_k,
    n_1,
    n_2,
)
params = fn_mol.init_params()  # initialize the parameters
```


```python
# initialize the Hamiltonian
hamiltonian = Hamiltonian(fn_mol)
```


```python
n_step = 10  # number of steps in each sampling
step_std_per_dim = (
    0.02  # standard deviation of the Gaussian noise in each step for each dimension
)

# initalize the sampler
sampler = Sampler(fn_mol, n_step, step_std_per_dim)
```


```python
n_samples = 4096  # number of samples

# initialize the driver
nnqs = NNQS(sampler, hamiltonian, fn_mol, n_samples)
```


```python
n_iters = 7000  # number of iterations

# training the network
params, energies = nnqs.train(n_iters, params, optax.chain(optax.adam(learning_rate=0.01), optax.add_noise(0.01,0.55,0))
)
```


      0%|          | 0/7000 [00:00<?, ?it/s]


    /opt/homebrew/Caskroom/miniforge/base/envs/jax-env/lib/python3.9/site-packages/jax/_src/ops/scatter.py:87: FutureWarning: scatter inputs have incompatible types: cannot safely cast value from dtype=int32 to dtype=bool. In future JAX releases this will result in an error.
      warnings.warn("scatter inputs have incompatible types: cannot safely cast "
    /opt/homebrew/Caskroom/miniforge/base/envs/jax-env/lib/python3.9/site-packages/jax/_src/ops/scatter.py:87: FutureWarning: scatter inputs have incompatible types: cannot safely cast value from dtype=int32 to dtype=bool. In future JAX releases this will result in an error.
      warnings.warn("scatter inputs have incompatible types: cannot safely cast "



```python
expected_energy = -8.07050
```


```python
fig, ax = plt.subplots(dpi=150)

ax.plot(energies, label="FermiNet")
ax.set_xlabel("iteration")
ax.set_ylabel("<E>")
ax.axhline(expected_energy, color="r", linestyle="--", label="true energy")
ax.legend()
ax.patch.set_facecolor("white")
fig.patch.set_facecolor("white")
```


    
![png](README_files/README_18_0.png)
    

