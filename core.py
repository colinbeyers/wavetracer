# core.py
import numpy as np
from .parameters import validate_params, categorize_wave_regime, k_from_lam, lam_from_k
from .excursion_vectors import (
    xi_shallow_first, zeta_shallow_first, xi_intermediate_first, zeta_intermediate_first, xi_deep_first, zeta_deep_first,
    xi_shallow_second, zeta_shallow_second, xi_intermediate_second, zeta_intermediate_second, xi_deep_second, zeta_deep_second
)


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
    
    trajectory = np.zeros((len(t), 2))  # Shape (num_timesteps, 2) for (x, z)
    
    # Define mappings for wave regime and order
    wave_functions = {
        "shallow": {"first": (xi_shallow, zeta_shallow), "second": (xi_shallow, zeta_shallow)},
        "intermediate": {"first": (xi_intermediate, zeta_intermediate), "second": (xi_intermediate, zeta_intermediate)},
        "deep": {"first": (xi_deep, zeta_deep), "second": (xi_deep, zeta_deep)},
    }

    # Ensure the order parameter is valid
    if order not in ["first", "second"]:
        raise ValueError("Invalid order. Must be 'first' or 'second'.")

    # Get the correct functions based on the wave regime and order
    xi_func, zeta_func = wave_functions[wave_regime][order]

    # Compute trajectory using the selected functions
    xi = xi_func(a, k, H, omega, t, x_0, z_0)
    zeta = zeta_func(a, k, H, omega, t, x_0, z_0)

    # Store results in a trajectory array
    trajectory = np.zeros((len(t), 2))
    trajectory[:, 0] = xi  # x positions
    trajectory[:, 1] = zeta  # z positions

    
    trajectory[:, 0] = x_p # Assign x positions
    trajectory[:, 1] = z_p  # Assign z positions

    
    return trajectory
