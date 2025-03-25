import math
import numpy as np

g = 9.81 # m/s

def k_from_lam(lam):
    return 2 * np.pi / lam

def lam_from_k(k):
    return 2 * np.pi / k

def wave_disp_relation(k, H, wave_regime):
    if wave_regime is "shallow":
        omega = math.sqrt(g*k)
    elif wave_regime is "intermediate":
        omega = math.sqrt(g*k*np.tanh(k*H))
    else:
        omega = k*math.sqrt(g*H)

    return omega