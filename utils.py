import math

g = 9.81 # m/s

def k_from_lam(lam):
    return 2 * math.pi / lam

def lam_from_k(k):
    return 2 * math.pi / k

def wave_disp_relation(k, H, wave_regime):
    if wave_regime is "shallow":
        omega = math.sqrt(g*k)
    elif wave_regime is "intermediate":
        omega = math.sqrt(g*k*math.tanh(k*H))
    else:
        omega = k*math.sqrt(g*H)

    return omega

def cfl(omega):
    T = 2 * math.pi / omega
    return T / 24