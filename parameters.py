import numpy as np
import math

def validate_params(a=None, H=None, k=None, lam=None, order='leading'):
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
    - order: "leading" or "first" order calculation.

    Raises:
    - ValueError: If any parameter is invalid or missing.
    - TypeError: If any parameter is not the correct type.
    """
    
    # Check if a is provided and valid
    if a is None:
        raise ValueError("Wave amplitude (a) must be provided and positive.")
    if not isinstance(a, (int, float)) or a <= 0:
        raise ValueError("Wave amplitude (a) must be a positive number [m].")

    # Check to see if at least lam or k is provided
    if k is None and lam is None:
        raise ValueError("At least wavelength (lambda) or wavenumber (k) must be provided.")

    # Check if lam is valid
    if lam is not None:
        if not isinstance(lam, (int, float)) or lam <= 0:
            raise ValueError("Wavelength (lambda) must be a positive number [m].")
    
    # Check if k is valid
    if k is not None:
        if not isinstance(k, (int, float)) or k <= 0:
            raise ValueError("Wavenumber (k) must be a positive number [1/m].")
    
    # If both lam and k are provided, check that they satisfy their relationship
    if lam is not None and k is not None:
        if not math.isclose(k, 2 * math.pi / lam, rel_tol=1e-5):
            raise ValueError(f"k = {k} and lambda = {lam} do not satisfy the relation k = 2π/λ.")

    # Check if H is provided and valid
    if H is None:
        raise ValueError("Water depth (H) must be provided and positive.")
    if not isinstance(H, (int, float)) or H <= 0:
        raise ValueError("Water depth (H) must be a positive float.")
    
    if order not in ('leading', 'first'):
        raise ValueError("Approximation order (order) must be either 'leading' or 'first'.")
    
    if not (a < 0.1 * lam):
        raise ValueError(f"Equations are only valid when a << λ. Condition is not satisfied for a = {a}, λ = {lam}.")

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
        regime = 'shallow'
    elif k * H > 3:  # Deep water condition
        regime = 'deep'
    else:  # Intermediate water condition
        regime = 'intermediate'

    return regime

def k_from_lam(lam):
    return 2 * np.pi / lam

def lam_from_k(k):
    return 2 * np.pi / k

def wave_disp_relation(k, H, wave_regime):
    g = 9.81 # m/s

    if wave_regime is 'shallow':
        omega = math.sqrt(g*k)
    elif wave_regime is 'intermediate':
        omega = math.sqrt(g*k*np.tanh(k*H))
    elif wave_regime is 'deep':
        omega = k*math.sqrt(g*H)

    return omega
