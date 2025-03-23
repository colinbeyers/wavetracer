import numpy as np
import math
from typing import Union

def xi_shallow(a: float, k: float, H: float, omega: float, t: Union[float, np.ndarray], 
               x_0: float, z_0: float) -> Union[float, np.ndarray]:
    """
    Calculate the horizontal excursion vector (xi) for shallow water waves.

    Args:
        a (float): Amplitude of the wave.
        k (float): Wave number.
        H (float): Water depth.
        omega (float): Angular frequency of the wave.
        t (Union[float, np.ndarray]): Time or an array of time values.
        x_0 (float): Initial horizontal position.
        z_0 (float): Initial vertical position (depth).

    Returns:
        Union[float, np.ndarray]: Horizontal excursion vector (xi) at time(s) t.
    """
    return -a * (1 / (k * H)) * np.sin(k * x_0 - omega * t)


def zeta_shallow(a: float, k: float, H: float, omega: float, t: Union[float, np.ndarray], 
                 x_0: float, z_0: float) -> Union[float, np.ndarray]:
    """
    Calculate the vertical excursion vector (zeta) for shallow water waves.

    Args:
        a (float): Amplitude of the wave.
        k (float): Wave number.
        H (float): Water depth.
        omega (float): Angular frequency of the wave.
        t (Union[float, np.ndarray]): Time or an array of time values.
        x_0 (float): Initial horizontal position.
        z_0 (float): Initial vertical position (depth).

    Returns:
        Union[float, np.ndarray]: Vertical excursion vector (zeta) at time(s) t.
    """
    return a * (z_0 / H + 1) * np.cos(k * x_0 - omega * t)


def xi_intermediate(a: float, k: float, H: float, omega: float, t: Union[float, np.ndarray], 
                   x_0: float, z_0: float) -> Union[float, np.ndarray]:
    """
    Calculate the horizontal excursion vector (xi) for intermediate water waves.

    Args:
        a (float): Amplitude of the wave.
        k (float): Wave number.
        H (float): Water depth.
        omega (float): Angular frequency of the wave.
        t (Union[float, np.ndarray]): Time or an array of time values.
        x_0 (float): Initial horizontal position.
        z_0 (float): Initial vertical position (depth).

    Returns:
        Union[float, np.ndarray]: Horizontal excursion vector (xi) at time(s) t.
    """
    return -a * (np.cosh(k * (z_0 + H)) / np.sinh(k * H)) * np.sin(k * x_0 - omega * t)


def zeta_intermediate(a: float, k: float, H: float, omega: float, t: Union[float, np.ndarray], 
                     x_0: float, z_0: float) -> Union[float, np.ndarray]:
    """
    Calculate the vertical excursion vector (zeta) for intermediate water waves.

    Args:
        a (float): Amplitude of the wave.
        k (float): Wave number.
        H (float): Water depth.
        omega (float): Angular frequency of the wave.
        t (Union[float, np.ndarray]): Time or an array of time values.
        x_0 (float): Initial horizontal position.
        z_0 (float): Initial vertical position (depth).

    Returns:
        Union[float, np.ndarray]: Vertical excursion vector (zeta) at time(s) t.
    """
    return a * (np.sinh(k * (z_0 + H)) / np.sinh(k * H)) * np.cos(k * x_0 - omega * t)


def xi_deep(a: float, k: float, H: float, omega: float, t: Union[float, np.ndarray], 
           x_0: float, z_0: float) -> Union[float, np.ndarray]:
    """
    Calculate the horizontal excursion vector (xi) for deep water waves.

    Args:
        a (float): Amplitude of the wave.
        k (float): Wave number.
        H (float): Water depth.
        omega (float): Angular frequency of the wave.
        t (Union[float, np.ndarray]): Time or an array of time values.
        x_0 (float): Initial horizontal position.
        z_0 (float): Initial vertical position (depth).

    Returns:
        Union[float, np.ndarray]: Horizontal excursion vector (xi) at time(s) t.
    """
    return -a * math.exp(k * z_0) * np.sin(k * x_0 - omega * t)


def zeta_deep(a: float, k: float, H: float, omega: float, t: Union[float, np.ndarray], 
             x_0: float, z_0: float) -> Union[float, np.ndarray]:
    """
    Calculate the vertical excursion vector (zeta) for deep water waves.

    Args:
        a (float): Amplitude of the wave.
        k (float): Wave number.
        H (float): Water depth.
        omega (float): Angular frequency of the wave.
        t (Union[float, np.ndarray]): Time or an array of time values.
        x_0 (float): Initial horizontal position.
        z_0 (float): Initial vertical position (depth).

    Returns:
        Union[float, np.ndarray]: Vertical excursion vector (zeta) at time(s) t.
    """
    return a * math.exp(k * z_0) * np.cos(k * x_0 - omega * t)
