import sys, os, h5py, time

import numpy as np

import readgadget
import readfof
import MAS_library as MASL
import Pk_library as PKL

import matplotlib.pyplot as plt
from pylab import *

# Parameters for Pylians
grid     = 512    # the density field will have grid^3 voxels
MAS      = 'CIC'  # Mass-assignment scheme: 'CIC'
verbose  = False  # whether to print information about the progress
ptype    = [1]    # [1] for CDM
axis     = 0      # axis along which RSD have been placed. In real-space this p>
threads  = 4      # number of openmp threads to compute the power spectrum
BoxSize  = 500   # Simulation volume is 1 Gpc.

def sort_pos(path_to_snaps_i, path_to_snaps_f, Quijote=True):
    """
    --------------------------------------------------------------------------
    Function: sort_pos
    Description:
        Reads initial and final particle data from a simulation snapshot file,
        sorts particles by ID, assigns them to a regular 3D grid, and computes
        the displacement field (difference between final and initial positions),
        accounting for periodic boundary conditions.

    Args:
        path_to_snaps_i : str
            Path to the initial condition snapshot.
        path_to_snaps_f : str
            Path to the final snapshot.
        Quijote : bool
            If True, assumes Quijote units (kpc/h and 1-based IDs),
            applies necessary corrections (ID-1, /BoxSize).
            If False, assumes COLA-style units and IDs.

     Returns:
        pos_f : ndarray
            Final positions of particles, sorted and mapped to the grid.
        pos_lag : ndarray
            Lagrangian (initial grid cell) positions of the particles.
        disp : ndarray
            Displacement vector from initial to final position for each particl>
    --------------------------------------------------------------------------
    """


    if Quijote:
        # Quijote snapshots: use 1-based IDs and positions in kpc/h
        ids_i = readgadget.read_block(path_to_snaps_i, "ID  ", ptype=[1]) - 1
        pos_i = readgadget.read_block(path_to_snaps_i, "POS ", ptype=[1]) / BoxSize
        pos_i = pos_i[np.argsort(ids_i)]

        ids_f = readgadget.read_block(path_to_snaps_f, "ID  ", ptype=[1]) - 1
        pos_f = readgadget.read_block(path_to_snaps_f, "POS ", ptype=[1]) / BoxSize
        pos_f = pos_f[np.argsort(ids_f)]
    else:
        # COLA snapshots: use 0-based IDs and positions already in Mpc/h
        ids_i = readgadget.read_block(path_to_snaps_i, "ID  ", ptype=[1])
        pos_i = readgadget.read_block(path_to_snaps_i, "POS ", ptype=[1])
        pos_i = pos_i[np.argsort(ids_i)]

        ids_f = readgadget.read_block(path_to_snaps_f, "ID  ", ptype=[1])
        pos_f = readgadget.read_block(path_to_snaps_f, "POS ", ptype=[1])
        pos_f = pos_f[np.argsort(ids_f)]


 # Assign each initial position to a grid cell
    grid_index_3D = (np.round((pos_i / BoxSize) * grid, decimals=0)).astype(np.int32)
    grid_index_3D[np.where(grid_index_3D == grid)] = 0  # wrap around periodic box edges

    # Convert to Lagrangian positions (center of the grid cell)
    pos_lag = grid_index_3D * BoxSize / grid

    # Flatten 3D index to 1D for sorting
    grid_index_1D = (grid_index_3D[:, 0] * grid**2 + grid_index_3D[:, 1] * grid + grid_index_3D[:, 2]
    )
    indexes_lag = np.argsort(grid_index_1D)

    # Sort final positions and lagrangian positions according to grid index
    pos_f = pos_f[indexes_lag]
    pos_lag = pos_lag[indexes_lag]

    return pos_lag, pos_f

def Pk(path_to_snaps=None, positions=None, Quijote=True):
    '''
    Read the path to the snap, or the positions, and compute the monopole of
    the power spectrum. It takes into account if the snaps were
    created using Quijote or COLA.
    Args:
        path_to_snaps : str or None
            Path to the snap.
        positions : ndarray or None
            N×3 array containing the positions [in Mpc/h] of N particles.
        Quijote : bool
            If path_to_snaps is not None, this flag indicates wether units are kpc/h (True) or Mpc/h (False).
    '''

    if positions is None and path_to_snaps is None:
        raise ValueError("You should provide a path to the snap or an array with the positions.")

    if positions is None:
        # Read particle positions from snap
        pos = readgadget.read_block(path_to_snaps, "POS ", ptype)
        if Quijote:
            pos = pos / BoxSize  # from kpc/h to Mpc/h

    else:
        # Use position directly
        pos = positions

    # Construct 3D density field
    delta = np.zeros((grid, grid, grid), dtype = np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)
    
    # compute the overdensity field
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0

    # compute power spectrum
    compute_Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads, verbose)

    # 3D P(k)
    k0 = compute_Pk.k3D           # Modes
    Pk0 = compute_Pk.Pk[:, 0]     # Monopole

    return k0, Pk0

def plot_Pk(k, Pk, linestyle='-', color='black', label='Pk_label', ax=None):
    '''
    It takes {k, P(k)} and plot it.
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,5))

    # Plot in log-log space
    ax.loglog(k, Pk, linestyle, color=color, label=label)

    # To see only the relevant scales
    ax.set_xlim(1e-2, 1.7)
    ax.set_ylim(1e2, 1e5)

    ax.set_xlabel(r'$k\ [h/{\rm Mpc}]$', fontsize=12)
    ax.set_ylabel(r'$P(k)\ [{\rm Mpc}/h]^3$', fontsize=12)
    
    ax.legend(loc='lower left')
    return ax

def plot_Tk(k, Pk_cola, Pk_quijote, linestyle='-', color='black', label='Pk_label', ax=None):
    # k_cola = k_quijote --> we just need one of them
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))

    # Plot in log-log space
    ax.semilogx(k, np.sqrt(Pk_cola/Pk_quijote), linestyle, color=color, label=label)
    ax.semilogx(k, np.ones(len(k)), '--', color='black')

    # To see only the relevant scales
    ax.set_xlim(1e-2, 1.7)
    ax.set_ylim(0.9, 1.05)

    ax.set_xlabel(r'$k\ [h/{\rm Mpc}]$', fontsize=10)
    ax.set_ylabel(r'$T(k)$', fontsize=10)

    ax.legend(loc='lower left')
    return ax
