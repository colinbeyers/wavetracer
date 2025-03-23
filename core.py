# core.py
import numpy as np
from .parameters import validate_params, categorize_wave_regime, k_from_lam, lam_from_k
from .excursion_vectors import xi_shallow, zeta_shallow, xi_intermediate, zeta_intermediate, xi_deep, zeta_deep

def compute_trajectory(a, H, t, x_0, z_0, k=None, lam=None, order="first"):
    """
    Computes the trajectory of a particle based on wave parameters.

    Parameters:
    - a: Wave amplitude (float, meters). Required.
    - k: Wavenumber (float, 1/m). Must be positive. Optional if lam is provided.
    - lam: Wavelength (float, meters). Must be positive. Optional if k is provided.
    - H: Water depth (float, meters). Required.
    - omega: Angular frequency (float, rad/s). Required.
    - t: Time(s) (float or array of floats). Required.
    - x_0: Initial x position of the particle (float, meters). Required.
    - z_0: Initial z position of the particle (float, meters). Required.

    Returns:
    - trajectory: Array of particle positions over time.

    Raises:
    - ValueError: If any of the input parameters are invalid.
    """
    
    # Validate the parameters
    validate_params(a=a, H=H, t=t, x_0=x_0, z_0=z_0, order=order, k=k, lam=lam)
    
    # Calculate wavenumber if wavelength is provided (or vice versa)
    if lam is not None and k is None:
        k = k_from_lam(lam)  # k from lam
    elif k is not None and lam is None:
        lam = lam_from_k(k)  # lam from k
    
    # Determine the wave regime based on the water depth and wavenumber
    wave_regime = categorize_wave_regime(k, H)
    
    # Initialize trajectory array
    if isinstance(t, (int, float)):  # if t is a single value
        t = np.array([t])
    
    trajectory = np.zeros_like(t)
    
    # Compute the particle trajectory based on the wave regime and approximations
    if wave_regime == "shallow":
        # Use the shallow water approximations
        xi = xi_shallow(a, k, H, omega, t, x_0, z_0)
        zeta = zeta_shallow(a, k, H, omega, t, x_0, z_0)
    elif wave_regime == "intermediate":
        # Use the intermediate water approximations
        xi = xi_intermediate(a, k, H, omega, t, x_0, z_0)
        zeta = zeta_intermediate(a, k, H, omega, t, x_0, z_0)
    else:  # deep water
        # Use the deep water approximations
        xi = xi_deep(a, k, H, omega, t, x_0, z_0)
        zeta = zeta_deep(a, k, H, omega, t, x_0, z_0)
    
    # The trajectory is the combination of xi and zeta
    trajectory = xi + zeta
    
    return trajectory
