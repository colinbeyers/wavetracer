# core.py
import numpy as np
import xarray as xr

from .parameters import validate_params, categorize_wave_regime, k_from_lam, lam_from_k, wave_disp_relation
from .waves import eta
from wavetracer.excursion_vectors.first_order_approx import (
    xi_shallow          as xi_shallow_first, 
    zeta_shallow        as zeta_shallow_first, 
    xi_intermediate     as xi_intermediate_first, 
    zeta_intermediate   as zeta_intermediate_first, 
    xi_deep             as xi_deep_first, 
    zeta_deep           as zeta_deep_first
)

from wavetracer.excursion_vectors.second_order_approx import (
    xi_shallow          as xi_shallow_second, 
    zeta_shallow        as zeta_shallow_second, 
    xi_intermediate     as xi_intermediate_second, 
    zeta_intermediate   as zeta_intermediate_second, 
    xi_deep             as xi_deep_second, 
    zeta_deep           as zeta_deep_second
)

# Define the function map globally at the module level
function_map = {
    "shallow": {
        "first": (xi_shallow_first, zeta_shallow_first),
        "second": (xi_shallow_second, zeta_shallow_second),
    },
    "intermediate": {
        "first": (xi_intermediate_first, zeta_intermediate_first),
        "second": (xi_intermediate_second, zeta_intermediate_second),
    },
    "deep": {
        "first": (xi_deep_first, zeta_deep_first),
        "second": (xi_deep_second, zeta_deep_second),
    },
}

def build_dataset(x_0, z_0, time, grid=False):
    """
    Builds an xarray Dataset with particle trajectories.

    Parameters:
    - x_0: Initial x positions of particles (array-like).
    - z_0: Initial z positions of particles (array-like).
    - time: Time array.
    - grid: If True, create a meshgrid for x_0 and z_0. If False, treat x_0 and z_0 as 1D arrays.

    Returns:
    - ds: xarray Dataset containing the particle data.
    """
    
    if grid:
        # Create a meshgrid of x_0 and z_0 if grid is True
        x_0, z_0 = np.meshgrid(x_0, z_0)
        # Flatten the meshgrid to create 1D arrays for particles
        x_0 = x_0.flatten()
        z_0 = z_0.flatten()
    else:
        # Ensure x_0 and z_0 are the same length
        if len(x_0) != len(z_0):
            raise ValueError("x_0 and z_0 must have the same length when grid=False.")
    
    # Build the Dataset with the provided coordinates and particle information
    ds = xr.Dataset(
        coords={
            "time": time,  # Time coordinate
            "particles": np.arange(len(x_0)),  # Particle index (0, 1, 2, ...)
            "x_0": ("particles", x_0),  # x_0 as a coordinate
            "z_0": ("particles", z_0),  # z_0 as a coordinate
        }
    )
    
    return ds

def compute_trajectories(ds, **kwargs):
    # Extract parameters from kwargs with default values
    a           = kwargs.get('a')
    H           = kwargs.get('H')
    k           = kwargs.get('k')
    lam         = kwargs.get('lam', None)
    order       = kwargs.get('order', "first")  # Default to "first" if not provided

    validate_params(a=a, H=H, k=k, lam=lam, order=order)
    
    # Calculate wavenumber if wavelength is provided (or vice versa)
    if lam is not None and k is None:
        k = k_from_lam(lam)  # k from lam
    elif k is not None and lam is None:
        lam = lam_from_k(k)  # lam from k
    
    # Determine the wave regime based on the water depth and wavenumber
    wave_regime = categorize_wave_regime(k, H)
    print("Regime: ", wave_regime)

    omega = wave_disp_relation(k=k, H=H, wave_regime=wave_regime)

    # Create a dictionary of wave parameters to pass as kwargs
    params = {
        'a': a,           # Wave amplitude
        'H': H,           # Water depth
        'k': k,           # Wavenumber
        'omega': omega,   # Angular frequency
        'order': order,   # Approximation order
        'wave_regime': wave_regime  # Wave regime
    }

    # Apply the trajectory calculation using kwargs
    ds['x'], ds['z'] = xr.apply_ufunc(
        compute_trajectory,
        ds['x_0'], ds['z_0'], ds['time'],
        kwargs=params,  # Pass the parameters as a dictionary of keyword arguments
        input_core_dims=[[], [], ['time']],
        output_core_dims=[['time'], ['time']],
        vectorize=True,  # Apply function across particles and time
        dask='allowed'  # Optional: Enable parallelism with Dask
    )

    min = int(ds.x.min())-1
    max = int(ds.x.max())+1

    x_eta = np.linspace(int(ds.x.min())-1, int(ds.x.max())+1, int((max-min)/lam * 32))
    print(x_eta)
    ds.coords['x_eta'] = x_eta  # Add 'x' as a coordinate to the dataset

    ds['eta'] = (('time', 'x_eta'), eta(x_eta, ds['time'].values, a, k, omega))

    return ds

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
