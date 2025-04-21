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

def random_mean_pos(N, x_start, x_end, H):
    x_0 = np.random.uniform(x_start, x_end, N)
    z_0 = np.random.uniform(-H, 0, N)
    return x_0, z_0

import numpy as np
import xarray as xr

def build_dataset(x_0=None, z_0=None, time=None, grid=False, random_pos=False, N=None, x_min=None, x_max=None, H_particles=None):
    """
    Builds an xarray Dataset with particle trajectories.

    Parameters:
    - x_0: Initial x positions of particles (array-like, required if random_pos=False).
    - z_0: Initial z positions of particles (array-like, required if random_pos=False).
    - time: Time array.
    - grid: If True, create a meshgrid for x_0 and z_0. If False, treat x_0 and z_0 as 1D arrays.
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
        # Generate random positions
        x_rand, z_rand = random_mean_pos(N, x_min, x_max, H_particles)
        
        # If x_0 and z_0 are provided, concatenate them with the random positions
        if x_0 is not None and z_0 is not None:
            if len(x_0) != len(z_0):
                raise ValueError("x_0 and z_0 must have the same length.")
            # Concatenate the provided x_0, z_0 with the random values
            x_0 = np.concatenate([x_0, x_rand])
            z_0 = np.concatenate([z_0, z_rand])
        else:
            # Only use the random positions
            x_0, z_0 = x_rand, z_rand

    else:
        # If random_pos is False, x_0 and z_0 must be provided
        if x_0 is None or z_0 is None:
            raise ValueError("When random_pos=False, x_0 and z_0 must be provided.")
        
        # Validate that x_0 and z_0 have the same length
        if len(x_0) != len(z_0):
            raise ValueError("x_0 and z_0 must have the same length when grid=False.")
        
        # If grid is True, create meshgrid and flatten
        if grid:
            x_0, z_0 = np.meshgrid(x_0, z_0)
            x_0 = x_0.flatten()
            z_0 = z_0.flatten()

    # Build the xarray Dataset
    ds = xr.Dataset(
        {
            "x_0": ("particle", x_0),
            "z_0": ("particle", z_0),
        },
        coords={
            "time": time,
            "particle": np.arange(len(x_0)),
        },
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
    # print("Regime: ", wave_regime)

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
    ds['x_p'], ds['z_p'] = xr.apply_ufunc(
        compute_trajectory,
        ds['x_0'], ds['z_0'], ds['time'],
        kwargs=params,  # Pass the parameters as a dictionary of keyword arguments
        input_core_dims=[[], [], ['time']],
        output_core_dims=[['time'], ['time']],
        vectorize=True,  # Apply function across particles and time
        dask='allowed'  # Optional: Enable parallelism with Dask
    )

    if order == 'second':
        min = int(ds.x_p.min())-1
        max = int(ds.x_p.max())+1
    else:
        min = int(ds.x_0.min())-5
        max = int(ds.x_0.max())+5

    x_eta = np.linspace(min, max, int((max-min)/lam * 32))
    # print(x_eta)
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
