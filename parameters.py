import numpy as np
import math

def validate_params(a, H, t, x_0, z_0, order, k=None, lam=None):
    """
    Validate the input parameters for the wave and particle calculations.

    Parameters:
    - a: Wave amplitude (float, meters). Must be positive.
    - k: Wavenumber (float, 1/m), or lam (float, meters). At least one of k or lam must be provided.
    - lam: Wavelength (float, meters). At least one of k or lam must be provided.
    - H: Water depth (float, meters). Must be positive.
    - t: Time(s) when particle position is calculated (float or array of floats, seconds).
    - x_0: Mean x position of particle (float, meters). Must be a float.
    - z_0: Mean z position of particle (float, meters). Must be a float.
    - order: "first" or "second" order calculation.

    Raises:
    - ValueError: If any parameter is invalid or missing.
    - TypeError: If any parameter is not the correct type.
    """
    
    # Check if a is provided and valid
    if a is None:
        raise ValueError("Wave amplitude (a) must be provided and positive.")
    if not isinstance(a, (int, float)) or a <= 0:
        raise ValueError("Wave amplitude (a) must be a positive float.")

    # Ensure at least one of k or lam is provided
    if k is None and lam is None:
        raise ValueError("Either wavenumber (k) or wavelength (lambda) must be provided.")

    # If lam is provided, calculate k later
    if lam is not None and not isinstance(lam, (int, float)) or lam <= 0:
        raise ValueError("Wavelength (lambda) must be a positive float.")
    
    # If k is provided, calculate lam later
    if k is not None and not isinstance(k, (int, float)) or k <= 0:
        raise ValueError("Wavenumber (k) must be a positive float (1/m).")
    
    # If both k and lam are provided, check that they satisfy the relation k = 2*pi/lam
    if k is not None and lam is not None:
        if not math.isclose(k, 2 * math.pi / lam, rel_tol=1e-5):
            raise ValueError(f"Provided k and lambda do not satisfy the relation k = 2*pi/lam. k = {k}, lambda = {lam}")

    # Check if H is provided and valid
    if H is None:
        raise ValueError("Water depth (H) must be provided and positive.")
    if not isinstance(H, (int, float)) or H <= 0:
        raise ValueError("Water depth (H) must be a positive float.")

    # Check if t is provided and valid (either a single float or an array of floats)
    if t is None:
        raise ValueError("Time (t) must be provided and valid.")
    if not isinstance(t, (int, float, np.ndarray)):
        raise TypeError("Time (t) must be a float or array of floats.")
    if isinstance(t, (int, float)) and t < 0:
        raise ValueError("Time (t) must be non-negative.")
    elif isinstance(t, np.ndarray) and np.any(t < 0):
        raise ValueError("All time values in array must be non-negative.")

    # Check if x_0 is provided and valid
    if x_0 is None:
        raise ValueError("Mean x position (x_0) must be provided and valid.")
    if not isinstance(x_0, (int, float)):
        raise TypeError("Mean x position (x_0) must be a float.")
    
    # Check if z_0 is provided and valid
    if z_0 is None:
        raise ValueError("Mean z position (z_0) must be provided and valid.")
    if not isinstance(z_0, (int, float)):
        raise TypeError("Mean z position (z_0) must be a float.")
    
    if order not in ["first", "second"]:
        raise ValueError("Parameter 'order' must be either 'first' or 'second'.")

    # Return True if everything is valid (optional, since no return is required)
    return True


def categorize_wave_regime(k, H):
    """
    Categorizes the wave regime based on the wavenumber (k) and water depth (H).

    The function classifies the wave as one of the following:
    - 'shallow' if the wave is in shallow water (k * H < 0.1)
    - 'deep' if the wave is in deep water (k * H > 3)
    - 'intermediate' if the wave is in intermediate water (0.1 <= k * H <= 3)

    Parameters:
    - k: Wavenumber of the wave (float, 1/m). Must be positive.
    - H: Water depth (float, meters). Must be positive.

    Returns:
    - A string: 'deep', 'shallow', or 'intermediate' depending on the wave regime.
    """

    if k * H < 0.1:  # Shallow water condition
        return "shallow"
    elif k * H > 3:  # Deep water condition
        return "deep"
    else:  # Intermediate water condition
        return "intermediate"

def k_from_lam(lam):
    return 2 * np.pi / lam

def lam_from_k(k):
    return 2 * np.pi / k