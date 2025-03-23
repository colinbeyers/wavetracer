import numpy as np
import math

def xi_shallow(a, k, H, omega, t, x_0, z_0):
    return -a * (1/(k*H)) * np.sin(k*x_0 - omega*t)

def zeta_shallow(a, k, H, omega, t, x_0, z_0):
    return a * (z_0/H + 1) * np.cos(k*x_0 - omega*t)

def xi_intermediate(a, k, H, omega, t, x_0, z_0):
    return -a * (np.cosh(k*(z_0+H))/np.sinh(k*H)) * np.sin(k*x_0 - omega*t)

def zeta_intermediate(a, k, H, omega, t, x_0, z_0):
    return a * (np.sinh(k*(z_0+H))/np.sinh(k*H)) * np.cos(k*x_0 - omega*t)

def xi_deep(a, k, H, omega, t, x_0, z_0):
    return -a * math.exp(k*z_0) * np.sin(k*x_0 - omega*t)

def zeta_deep(a, k, H, omega, t, x_0, z_0):
    return a * math.exp(k*z_0) * np.cos(k*x_0 - omega*t)