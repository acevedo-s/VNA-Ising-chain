# VNA-Ising-chain

'RNN_1D_Ising.ipynb' starts with the analytical calculation of the partition function and the two-point correlation function of the 1D ferromagnetic Ising chain, via transfer matrix.

The code below calculates the entropy, free energy and the two-point correlator of the system at finite temperature, using the technique known as 'variational neural annealing'.
To know how this algorithm works in full detail, see https://arxiv.org/pdf/2101.10154.pdf \\
All routins used in 'RNN_1D_Ising.ipynb' are in 'VNA.py'.

Exact and numerical results are compared, finding good agreement.
