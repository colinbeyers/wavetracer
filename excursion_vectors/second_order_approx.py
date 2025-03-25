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
    L1 = 1/(k*H)
    L2 = (z_0/H + 1)
    phi = k*x_0 - omega*t

    xi_2_shlw = -a*L1*np.sin(phi) + a**2*omega*k/2*(L1**2+L2**2)*t + a**2*k/4*(L2**2-L1**2)*np.sin(2*phi)

    return xi_2_shlw


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
    L2 = (z_0/H + 1)
    phi = k*x_0 - omega*t

    zeta_2_shlw = a*L2*np.cos(phi)

    return zeta_2_shlw


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
    L1 = math.cosh(k*(z_0+H)) / math.sinh(k*H)
    L2 = math.sinh(k*(z_0+H)) / math.sinh(k*H)
    phi = k*x_0 - omega*t

    xi_2_intm = -a*L1*np.sin(phi) + a**2*omega*k/2*(L1**2+L2**2)*t + a**2*k/4*(L2**2-L1**2)*np.sin(2*phi)

    return xi_2_intm


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
    L2 = math.sinh(k*(z_0+H)) / math.sinh(k*H)
    phi = k*x_0 - omega*t

    zeta_2_intm = a*L2*np.cos(phi)

    return zeta_2_intm


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
    L1 = math.exp(k*z_0)
    phi = k*x_0 - omega*t

    xi_2_deep = -a*L1*np.sin(phi) + a**2*omega*k*L1**2*t

    return xi_2_deep


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
    L2 = math.exp(k*z_0)
    phi = k*x_0 - omega*t

    zeta_2_deep = a*L2*np.cos(phi)

    return zeta_2_deep