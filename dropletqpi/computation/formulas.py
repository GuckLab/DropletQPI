import numpy as np


def interpolate_lin(a, c, x):
    return a*x + c


def calculate_sphere_vol(r):
    return 4/3 * np.pi * r**3


def calculate_2D_dist(x0, y0, x1, y1):
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)


def calculate_drymass_density(n_s, n_m, alpha=0.19):
    return (n_s - n_m) / alpha


def calculate_refr_increment(n_s, n_m, rho):
    return (n_s - n_m) / rho


def calculate_absolute_density(rho_dry, rho_fluid, part_spec_vol_prot=0.73):
    # m_dry = rho_dry * volume
    return rho_dry + rho_fluid - (rho_fluid * rho_dry * part_spec_vol_prot)
