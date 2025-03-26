import numpy as np

def eta(x, t, a, k, omega):
    """
    Computes waveform at each t across space x.
    """

    if isinstance(t, (int, float)):
        t = np.array([t])
    
    eta = a*np.cos(k * x[np.newaxis, :] - omega * t[:, np.newaxis])

    return eta