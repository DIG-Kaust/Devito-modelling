import numpy as np
import os
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import Tuple
from typing import Union
from tqdm import tqdm
import warnings

from devito import *
from examples.seismic.elastic.wavesolver import ElasticWaveSolver
from examples.seismic.utils import AcquisitionGeometry
from examples.seismic import Model
from examples.seismic.source import TimeAxis
from examples.seismic.source import RickerSource, Receiver, TimeAxis
from sympy import init_printing, latex
init_printing(use_latex='mathjax')

class Elastic2D():
    def __init__(self):
        pass

    def create_model(self, vp, vs, rho, shape, origin, spacing, space_order: int=6, nbl: int=20, fs: bool=False, seismic_model=None):
        """Create model

        Parameters
        ----------
        vp : :obj:`numpy.ndarray`
            P-wave Velocity model in km/s
        vs : :obj:`numpy.ndarray`
            S-wave Velocity model in km/s
        rho : :obj:`numpy.ndarray`
            Density model in kg/m^3
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries
        fs : :obj:`bool`, optional
            Add free surface

        """

        self.space_order = space_order
        self.fs = fs
        self.vp = vp
        self.vs = vs
        self.spacing = spacing
        self.model = Model(vp=vp, vs=vs, b=1/rho, origin=origin, shape=shape, spacing=spacing, 
                           dtype=np.float32, space_order=space_order, nbl=nbl, bcs="mask", fs=fs)


    def create_geometry(self, src_x: npt.DTypeLike, src_z: npt.DTypeLike, 
                        rec_x: npt.DTypeLike, rec_z: npt.DTypeLike, t0: float, tn: int, 
                        src_type: str=None, f0: float=60):
        """Create geometry and time axis

        Parameters
        ----------
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in m
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in m
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in m
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in m
        t0 : :obj:`float`
            Initial time 
        tn : :obj:`int`
            Number of time samples
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc,2))
        src_coordinates[:,0] = src_x
        src_coordinates[:,1] = src_z
        
        rec_coordinates = np.empty((nrec,2))
        rec_coordinates[:,0] = rec_x
        rec_coordinates[:,1] = rec_z

        

        self.geometry = AcquisitionGeometry(self.model, rec_coordinates, src_coordinates, 
                                            t0=t0, tn=tn, src_type=src_type, 
                                            f0=None if f0 is None else f0 * 1e-3,fs=self.model.fs)
        
        if (self.vp.min()*1000/f0) < self.spacing[0]*5 or (self.vp.min()*1000/f0) < self.spacing[1]*5 or (self.vs.min()*1000/f0) < self.spacing[0]*5 or (self.vs.min()*1000/f0) < self.spacing[1]*5:
            warnings.warn("To avoid numerical dispersion in your data, ensure that the following condition is met: min(v)/f0 < 5dx and min(v)/f0 < 5dz. Either refine the velocity grid or reduce the f0 frequency to mitigate this issue.")        

    def solve_one_shot(self, isrc, wav: npt.DTypeLike=None, dt: float=None, saveu: bool=False):
        """Solve wave equation for one shot

        Parameters
        ----------
        isrc : :obj:`float`
            Index of source to model
        wav : :obj:`float`, optional
            Wavelet (if not provided, use wavelet in geometry)
        dt : :obj:`float`, optional
            Time sampling of data (will be resampled)
        saveu : :obj:`bool`, optional
            Save snapshots

        Returns
        -------
        d1 : :obj:`np.ndarray`
            Data
        d2 : :obj:`np.ndarray`
            First derivative data
        v : :obj:`np.ndarray`
            The computed particle velocity snapshot
        tau : :obj:`np.ndarray`
            The computed symmetric stress tensor snapshot

        """
        #Create geometry for single source
        geometry = AcquisitionGeometry(self.model,self.geometry.rec_positions,self.geometry.src_positions[isrc,:],
                                       self.geometry.t0, self.geometry.tn, f0 = self.geometry.f0,
                                       src_type = self.geometry.src_type, fs=self.model.fs)

        src = None
        if wav is not None:
            # assign wavelet
            dt = self.model.critical_dt
            time_range = TimeAxis(start=self.geometry.t0, stop=self.geometry.tn, step=dt)
            src = RickerSource(name='src', grid=self.model.grid, f0=self.geometry.f0, time_range=time_range)
            src.coordinates.data[:, 0] = geometry.src.coordinates.data[isrc, 0]
            src.coordinates.data[:, 1] = geometry.src.coordinates.data[isrc, 1]
            src.data[:] = wav

        #Solve
        solver = ElasticWaveSolver(self.model, geometry, space_order=self.model.space_order)
        d1,d2,v,tau,_ = solver.forward(src=src,save=saveu)
        taxis = d1.time_values

        #Resample
        if dt is not None:
            d1 = d1.resample(dt)
            d2 = d1.resample(dt)
            taxis = d1.time_values

        return d1,d2,v,tau,taxis
        
    def solve_all_shots(self, wav: npt.DTypeLike = None, dt: float = None, tqdm_signature = None, 
                        figdir: str = None, datadir: str = None, savedtot: bool = False):
        """Solve wave equation for all shots in geometry

        Parameters
        ----------
        wav : :obj:`float`, optional
            Wavelet (if not provided, use wavelet in geometry)
        dt : :obj:`float`, optional
            Time sampling of data (will be resampled)
        tqdm_signature : :obj:`func`, optional
            tqdm function handle to use in for loop
        figdir : :obj:`bool`, optional
            Directory where to save figures for each shot
        datadir : :obj:`bool`, optional
            Directory where to save each shot in npz format
        savedtot : :obj:`bool`, optional
            Save total data

        Returns
        -------
        d1tot : :obj:`np.ndarray`
            Data
        d2tot : :obj:`np.ndarray`
            First derivative data
        taxis : :obj:`np.ndarray`
            Time axis

        """

        # Create figure directory
        if figdir is not None:
            if not os.path.exists(figdir):
                os.mkdir(figdir)

        # Create data directory
        if datadir is not None:
            if not os.path.exists(datadir):
                os.mkdir(datadir)

        # Model dataset (serial mode)
        nsrc = self.geometry.src_positions.shape[0]
        d1tot = []
        d2tot = []
        
        taxis = None
        if tqdm_signature is None:
            tqdm_signature = tqdm
        for isrc in tqdm_signature(range(nsrc)):

            d1, d2, _, _, _ = self.solve_one_shot(isrc, wav=wav, dt=dt)
            if isrc == 0:
                taxis = d1.time_values
            if savedtot:
                d1tot.append(d1.data)
                d2tot.append(d2.data)

            if datadir is not None:
                np.save(os.path.join(datadir, f'Shot{isrc}'), d.data)

            if figdir is not None:
                self.plot_shotrecord(d1.data, clip=1e-3, figpath=os.path.join(figdir, f'Shot{isrc}'))
                self.plot_shotrecord(d2.data, clip=1e-3, figpath=os.path.join(figdir, f'Shot{isrc}'))
                 
        # combine all shots in (s,r,t) cube
        if savedtot:
            d1tot = np.array(d1tot).transpose(0, 2, 1)
            d2tot = np.array(d2tot).transpose(0, 2, 1)
            
        return d1tot, d2tot, taxis

    def plot_velocity(self, source=True, receiver=True, colorbar=True, cmap="jet", figsize=(7, 7), figpath=None):
        """Display velocity model

        Plot a two-dimensional velocity field. Optionally also includes point markers for
        sources and receivers.

        Parameters
        ----------
        source : :obj:`bool`, optional
            Display sources
        receiver : :obj:`bool`, optional
            Display receivers
        colorbar : :obj:`bool`, optional
            Option to plot the colorbar
        cmap : :obj:`str`, optional
            Colormap
        figsize : :obj:`tuple`, optional
            Size of figure
        figpath : :obj:`str`, optional
            Full path (including filename) where to save figure

        """
        domain_size = 1.e-3 * np.array(self.model.domain_size)
        extent = [self.model.origin[0], self.model.origin[0] + domain_size[0],
                  self.model.origin[1] + domain_size[1], self.model.origin[1]]

        slices = list(slice(self.model.nbl, -self.model.nbl) for _ in range(2))
        if self.model.fs:
            slices[1] = slice(0, -self.model.nbl)
        if getattr(self.model, 'vp', None) is not None:
            field = self.model.vp.data[slices]
        else:
            field = np.sqrt((self.model.lam.data + 2*self.model.lam.data) * self.model.b.data)[slices]

        plt.figure(figsize=figsize)
        plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                          vmin=np.min(field), vmax=np.max(field),
                          extent=extent)
        plt.xlabel('X position (km)')
        plt.ylabel('Depth (km)')

        # Plot source points, if provided
        if receiver:
            plt.scatter(1e-3 * self.geometry.rec_positions[::5, 0], 1e-3 * self.geometry.rec_positions[::5, 1],
                        s=25, c='black', marker='D')

        # Plot receiver points, if provided
        if source:
            plt.scatter(1e-3 * self.geometry.src_positions[::5, 0], 1e-3 * self.geometry.src_positions[::5, 1],
                        s=25, c='red', marker='o')

        # Ensure axis limits
        plt.xlim(self.model.origin[0], self.model.origin[0] + domain_size[0])
        plt.ylim(self.model.origin[1] + domain_size[1], self.model.origin[1])

        # Create aligned colorbar on the right
        if colorbar==True:
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(plot, cax=cax)
            cbar.set_label('Velocity (km/s)')

        # Save figure
        if figpath:
            plt.savefig(figpath)


    def plot_shotrecord(self, rec, colorbar=True, clip=1, extent=None, figsize=(8, 8), figpath=None, cmap=cm.gray):
        """Plot a shot record (receiver values over time).


        Plot a two-dimensional velocity field. Optionally also includes point markers for
        sources and receivers.

        Parameters
        ----------
        rec : :obj:`np.ndarray`, optional
            Receiver data of shape (time, points).
        colorbar : :obj:`bool`, optional
            Option to plot the colorbar
        clip : :obj:`str`, optional
            Clipping
        figsize : :obj:`tuple`, optional
            Size of figure
        figpath : :obj:`str`, optional
            Full path (including filename) where to save figure

        """

        scale = np.max(rec) * clip
        # extent = [self.model.origin[0], self.model.origin[0] + 1e-3 * self.model.domain_size[0],
        #           1e-3 * self.geometry.tn, self.geometry.t0]
        extent = [self.model.origin[0], self.model.origin[0] + 1e-3 * self.model.domain_size[0],
                  1e-3 * self.geometry.time_axis.stop, 0]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot = ax.imshow(rec, vmin=-scale, vmax=scale, cmap=cmap, extent=extent)
        ax.axis('tight')
        ax.set_xlabel('X position (km)')
        ax.set_ylabel('Time (s)')

        # Create aligned colorbar on the right
        # if colorbar:
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     plt.colorbar(plot, cax=cax)

        # Save figure
        if figpath:
            plt.savefig(figpath)

        return ax
        



    
