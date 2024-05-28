######################################################################
# Acoustic modelling with Devito (distributed across shots with Ray)

# This notebook showcases how to perform 2D acoustic modelling Devito. 
# Ray is used to parallelize the computations across shots.
#
# Run as: export DEVITO_LANGUAGE=openmp; export OMP_NUM_THREADS=8; python PAcoustic2d.py

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import devito
import ray 

from scipy.interpolate import RegularGridInterpolator
from examples.seismic import Model, RickerSource

from devitomod.acoustic.ac2d import Acoustic2D

os.environ["RAY_DEDUP_LOGS"] = '0'
devito.configuration['log-level'] = 'ERROR'


# Initialize ray
ray.init(num_cpus=4, include_dashboard=False)


# Load Velocity model
f = np.load('../data/SeamPhase1.npz')

vp = f['vp']
x, z = f['x'], f['z']
nx, nz = len(x), len(z)
dx, dz = x[1] - x[0], z[1] - z[0]


# Modelling parameters

# velocity model parameter
shape = (nx, nz)
spacing = (dx, dz)
origin = (0, 0)

# source and receiver geometry
nsrc = 16
nrec = nx
osrc = x[-1]//4

# other modelling params
nbl = 150 # Number of boundary layers around the domain
space_order = 6 # Space order of the simulation
f0 = 10 # Source peak frequency (Hz)
fs = True
fslabel = '_fs' if fs else ''

t0 = 0. # Initial time
tmax = 5000 # Total simulation time (ms)
dt = 4 # Time sampling of observed data (ms)


# Modelling
awe = Acoustic2D()

awe.create_model(shape, origin, spacing, vp.T, space_order, nbl=nbl, fs=fs)
awe.create_geometry(src_x=np.arange(0, nsrc) * spacing[0] * 100 + osrc,
                    src_z=20,
                    rec_x=np.arange(0, nrec) * spacing[0],
                    rec_z=20,
                    t0=t0, tn=tmax, src_type='Ricker', f0=f0)

awe.plot_velocity(figsize=(17, 9))


# Model multiple shots in parallel
dtot, taxis = awe.ray_solve_all_shots();