# core.py
import numpy as np
import xarray as xr

from .parameters import validate_params, categorize_wave_regime, k_from_lam, lam_from_k, wave_disp_relation
from . import utils

from excursion_vectors.leading_order import (
    xi_shallow          as xi_shallow_0, 
    zeta_shallow        as zeta_shallow_0, 
    xi_intermediate     as xi_intermediate_0, 
    zeta_intermediate   as zeta_intermediate_0, 
    xi_deep             as xi_deep_0, 
    zeta_deep           as zeta_deep_0
)

from excursion_vectors.first_order import (
    xi_shallow          as xi_shallow_1, 
    zeta_shallow        as zeta_shallow_1, 
    xi_intermediate     as xi_intermediate_1, 
    zeta_intermediate   as zeta_intermediate_1, 
    xi_deep             as xi_deep_1, 
    zeta_deep           as zeta_deep_1
)

# Define the function map globally at the module level
function_map = {
    'shallow': {
        'leading': (xi_shallow_0, zeta_shallow_0),
        'first': (xi_shallow_1, zeta_shallow_1),
    },
    'intermediate': {
        'leading': (xi_intermediate_0, zeta_intermediate_0),
        'first': (xi_intermediate_1, zeta_intermediate_1),
    },
    'deep': {
        'leading': (xi_deep_0, zeta_deep_0),
        'first': (xi_deep_1, zeta_deep_1),
    },
}

def random_mean_positions(N, x_start, x_end, h):
    x_0 = np.random.uniform(x_start, x_end, N)
    z_0 = np.random.uniform(-h, 0, N)
    return x_0, z_0

def build_dataset(x0=None, z0=None, random_pos=False, N=None, x_min=None, x_max=None, H_particles=None, **kwargs):
    """
    Builds an xarray Dataset with mean particle positions and wave parameters.

    Parameters:
    - x_0: Initial x positions of particles (array-like, required if random_pos=False).
    - z_0: Initial z positions of particles (array-like, required if random_pos=False).
    - time: Time array.
    - random_pos: If True, generate random positions instead of using x_0, z_0.
    - N: Number of random particles (required if random_pos=True).
    - x_min: Start of x range for random positions (required if random_pos=True).
    - x_max: End of x range for random positions (required if random_pos=True).
    - H_particles: Maximum depth for random z positions (required if random_pos=True).

    Returns:
    - ds: xarray Dataset containing the particle data.
    """
    # If random_pos is True, generate random positions
    if random_pos:
        if None in [N, x_min, x_max, H_particles]:
            raise ValueError("When random_pos=True, N, x_min, x_max, and H_particles must be provided.")
        x_rand, z_rand = random_mean_positions(N, x_min, x_max, H_particles)
        
        # If x_0 and z_0 are provided, concatenate them with the random positions
        if x0 is not None and z0 is not None:
            if len(x0) != len(z0):
                raise ValueError("x_0 and z_0 must have the same length.")
            x0 = np.concatenate([x0, x_rand])
            z0 = np.concatenate([z0, z_rand])
        else:
            x0, z0 = x_rand, z_rand

    else:
        # If random_pos is False, x_0 and z_0 must be provided
        if x0 is None or z0 is None:
            raise ValueError("When random_pos=False, x_0 and z_0 must be provided.")
        
        # Validate that x_0 and z_0 have the same length
        if len(x0) != len(z0):
            raise ValueError("x0 and z0 must have the same length when grid=False.")

    # Build the xarray Dataset
    ds = xr.Dataset(
        {
            "x0": ("particle", x0),
            "z0": ("particle", z0),
        },
        coords={
            "particle": np.arange(len(x0)),
        },
    )

    return ds

def compute_trajectories(ds, duration, **kwargs):
    # Extract physical parameters from kwargs
    a       = kwargs.get('a', None)
    H       = kwargs.get('H', None)
    k       = kwargs.get('k', None)
    lam     = kwargs.get('lam', None)
    order   = kwargs.get('order', 'leading')

    validate_params(a=a, H=H, k=k, lam=lam, order=order)

    # Calculate k if lam is provided (or vice versa)
    if lam is not None and k is None:
        k = k_from_lam(lam)
    elif k is not None and lam is None:
        lam = lam_from_k(k)

    wave_regime = categorize_wave_regime(k, H)
    omega       = wave_disp_relation(k=k, H=H, wave_regime=wave_regime)

    # Calculate ideal dt and time array and add to dataset
    dt      = utils.cfl(omega)
    time    = np.arange(0, duration, dt)
    ds      = ds.assign_coords(time=("time", time))

    # Create a dictionary of wave parameters to pass as kwargs
    wave_params = {
        'a': a,
        'H': H,         
        'k': k,        
        'omega': omega,
        'order': order,
        'wave_regime': wave_regime
    }

    # Apply the trajectory calculation using kwargs
    ds['xp'], ds['zp'] = xr.apply_ufunc(
        compute_trajectory,
        ds['x0'], ds['z0'], ds['time'],
        kwargs=wave_params,  # Pass the parameters as a dictionary of keyword arguments
        input_core_dims=[[], [], ['time']],
        output_core_dims=[['time'], ['time']],
        vectorize=True,  # Apply function across particles and time
        dask='allowed'  # Optional: Enable parallelism with Dask
    )

    ds.attrs.update({
        'a': {'value': a, 'description': 'Wave amplitude (meters)'},
        'H': {'value': H, 'description': 'Water depth (meters)'},
        'k': {'value': k, 'description': 'Wavenumber (1/meters)'},
        'lam': {'value': lam, 'description': 'Wavelength (meters)'},
        'order': {'value': order, 'description': 'Approximation order ("leading" or "first")'}
    })

    ds_eta = compute_free_surface(ds, wave_params)

    x_eta_array = ds_eta['xp'].transpose('time', 'particle').data
    z_eta_array = ds_eta['zp'].transpose('time', 'particle').data

    ds = ds.assign(
        x_eta=(('time', 'point'), x_eta_array),
        z_eta=(('time', 'point'), z_eta_array)
    )

    return ds

def compute_free_surface(ds_p, wave_params):
    lam     = ds_p.attrs['lam']['value']
    time    = ds_p['time']

    x_min = int(ds_p.xp.min()) - 0.1 * lam
    x_max = int(ds_p.xp.max()) + 0.1 * lam

    x0_eta = np.linspace(x_min, x_max, int((x_max - x_min) / lam * 32))
    z0_eta = np.zeros_like(x0_eta)

    ds_eta = build_dataset(x0=x0_eta, z0=z0_eta, random_pos=False)
    ds_eta = ds_eta.assign_coords(time=("time", time.values))

    # calculate free surface particle trajectories
    ds_eta['xp'], ds_eta['zp'] = xr.apply_ufunc(
        compute_trajectory,
        ds_eta['x0'], ds_eta['z0'], ds_eta['time'],
        kwargs=wave_params,
        input_core_dims=[[], [], ['time']],
        output_core_dims=[['time'], ['time']],
        vectorize=True,
        dask='allowed'
    )

    return ds_eta

def compute_trajectory(x_0, z_0, t, **kwargs):
    """
    Computes the trajectory of a particle based on wave parameters.

    Parameters are passed via kwargs.
    
    Required kwargs:
    - a: Wave amplitude (float, meters).
    - k: Wavenumber (float, 1/m).
    - omega: Angular frequency (float, rad/s).
    - H: Water depth (float, meters).
    - order: The order for the approximation ("first", "second", etc.).
    - wave_regime: Type of wave regime ("intermediate", "shallow", etc.).

    Returns:
    - x_p: Particle x position over time.
    - z_p: Particle z position over time.
    """
    # Extract parameters from kwargs
    a           = kwargs.get('a')  # Wave amplitude
    k           = kwargs.get('k')  # Wavenumber
    H           = kwargs.get('H')  # Water depth
    omega       = kwargs.get('omega') # Wave Angular frequency
    order       = kwargs.get('order', "first")  # Default to "first" if not provided
    wave_regime = kwargs.get('wave_regime', "intermediate")  # Default to "intermediate"

    # Get the appropriate xi and zeta functions
    xi_func, zeta_func = function_map[wave_regime][order]

    # Compute the particle trajectory based on the wave regime and approximations
    xi = xi_func(a, k, H, omega, t, x_0, z_0)
    zeta = zeta_func(a, k, H, omega, t, x_0, z_0)

    # Compute paticle position
    x_p = x_0 + xi
    z_p = z_0 + zeta

    return x_p, z_p  # Return the particle positions over time
