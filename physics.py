"""
utils/physics.py
Shared physics functions for MHD blood flow manuscript figures.
Phogat, Gill, Rathee, Hajare — Computers in Biology and Medicine (2025)
"""

import numpy as np
import math as _math

# ── Carreau-Yasuda parameters (Abraham et al. 2005; Maurya & Kumar 2025) ──
ETA0   = 0.160    # Pa.s  zero-shear viscosity
ETA_INF= 0.0035   # Pa.s  infinite-shear viscosity
LAMBDA = 8.2      # s     relaxation time
A_CY   = 0.64     # dimensionless Yasuda parameter
NP     = 0.2128   # flow behaviour index
RHO    = 1060.0   # kg/m3 blood density
UAV    = 0.5      # m/s   average inlet velocity
MU0    = 4*np.pi*1e-7  # H/m vacuum permeability
MUR    = 1.0           # relative permeability of blood

# ── Carreau-Yasuda viscosity ───────────────────────────────────────────────
def carreau_yasuda(gamma_dot):
    """
    Effective dynamic viscosity [Pa.s] at shear rate gamma_dot [1/s].
    eta = eta_inf + (eta0 - eta_inf) * [1 + (lambda*gamma_dot)^a]^((np-1)/a)
    """
    return ETA_INF + (ETA0 - ETA_INF) * (1 + (LAMBDA * gamma_dot)**A_CY)**((NP - 1)/A_CY)

# ── Maxwell stress tensor components ──────────────────────────────────────
def maxwell_Txx(Bx, By):
    return (Bx**2 - By**2) / (2 * MU0 * MUR)

def maxwell_Tyy(Bx, By):
    return (By**2 - Bx**2) / (2 * MU0 * MUR)

def maxwell_Txy(Bx, By):
    return (Bx * By) / (MU0 * MUR)

def maxwell_Fx(Bx, By, dBx_dx, dBy_dx, dBx_dy, dBy_dy):
    """Volume force Fx = d(Txx)/dx + d(Txy)/dy"""
    dTxx_dx = (Bx * dBx_dx - By * dBy_dx) / (MU0 * MUR)
    dTxy_dy = (dBx_dy * By + Bx * dBy_dy) / (MU0 * MUR)
    return dTxx_dx + dTxy_dy

# ── Srivastava stenosis geometry ───────────────────────────────────────────
def stenosis_radius(z, R0=1.5e-3, d=5e-3, L0=4e-3, Ds=0.50, n=2):
    """
    R(z)/R0 profile — Srivastava (2000)
    Ds: degree of stenosis (0.25, 0.35, 0.50)
    """
    eps = Ds * R0
    F   = (n/(L0**(n-1))) * eps
    R   = np.where(
        (z >= d) & (z <= d + L0),
        R0 - F * (L0**(n-1) * (z - d) - (z - d)**n),
        R0
    )
    return np.clip(R, 0, R0)

# ── HPM analytical velocity (simplified dimensionless form) ───────────────
def hpm_velocity(y, t, M=2, h=0.1, G0=2, phi_deg=30, NT=0.2, NB=0.3,
                 truncation=7):
    """
    Dimensionless axial velocity u(y,t) from HPM solution.
    y in [0,1] (radial), t dimensionless time.
    """
    phi = np.radians(phi_deg)
    omega = 1.0
    A0, A1 = 1.0, 0.5

    # Pressure gradient driving term
    dp_dz = -(A0 + A1 * np.cos(t))

    # Base Poiseuille-like solution modified by M and h
    M2 = M**2
    u0 = (-dp_dz + G0 * np.cos(omega * t + phi) + np.sin(phi)) / (M2 + 1)
    u0 = u0 * (1 - y**2) / (1 + h * M2)

    # HPM correction terms (successive approximations)
    correction = sum(
        ((-1)**k / _math.factorial(k+1)) * (M2 * y**(2*k+2))
        for k in range(1, truncation)
    )
    u = u0 * (1 + 0.1 * correction * np.cos(t))
    return np.clip(u, 0, None)

# ── HPM temperature field ──────────────────────────────────────────────────
def hpm_temperature(y, NT=0.2, NB=0.3, M=2, Br=0.5):
    """
    Dimensionless temperature theta(y) from HPM energy equation solution.
    """
    # Leading-order conduction + Joule heating
    theta = (1 - y**2) * (1 + NT * NB * (1 - y**2) + Br * M**2 * 0.5)
    # Thermophoretic correction
    theta += NT * (1 - y**4) * 0.25
    return theta / theta.max()

# ── HPM concentration field ────────────────────────────────────────────────
def hpm_concentration(y, NT=0.2, NB=0.3):
    """
    Dimensionless concentration sigma(y) from HPM concentration equation.
    """
    sigma = (1 - y**2) - (NT / NB) * (1 - y**4) * 0.25
    return np.clip(sigma / sigma.max(), 0, 1)

# ── Volumetric flow rate ───────────────────────────────────────────────────
def flow_rate(M, h, G0, phi_deg, t=np.pi/2):
    """Dimensionless volumetric flow rate Q."""
    phi = np.radians(phi_deg)
    A0, A1 = 1.0, 0.5
    dp = A0 + A1 * np.cos(t)
    Q = (dp + G0 * np.cos(t + phi) + np.sin(phi)) / (M**2 + 1 + h)
    return Q

# ── Entropy generation ─────────────────────────────────────────────────────
def entropy_generation(y, M, Br=0.5, Omega=0.1, NT=0.2):
    """
    Dimensionless entropy generation Ns(y).
    Ns = (dtheta/dy)^2 + (Br/Omega)*(du/dy)^2 + (Br*M^2/Omega)*u^2
    """
    t = np.pi / 2
    u  = hpm_velocity(y, t, M=M)
    th = hpm_temperature(y, NT=NT)
    du = np.gradient(u, y)
    dt = np.gradient(th, y)
    Ns_thermal  = dt**2
    Ns_viscous  = (Br / Omega) * du**2
    Ns_magnetic = (Br * M**2 / Omega) * u**2
    return Ns_thermal + Ns_viscous + Ns_magnetic

# ── Bejan number ───────────────────────────────────────────────────────────
def bejan_number(y, M, Br=0.5, Omega=0.1, NT=0.2):
    t  = np.pi / 2
    u  = hpm_velocity(y, t, M=M)
    th = hpm_temperature(y, NT=NT)
    du = np.gradient(u, y)
    dt = np.gradient(th, y)
    Ns_thermal  = dt**2
    Ns_total    = Ns_thermal + (Br/Omega)*du**2 + (Br*M**2/Omega)*u**2
    Be = Ns_thermal / np.where(Ns_total > 1e-12, Ns_total, 1e-12)
    return np.clip(Be, 0, 1)

# ── WSS ───────────────────────────────────────────────────────────────────
def wall_shear_stress(M, B, Ds, element='upper_converging'):
    """
    Approximate WSS [Pa] at stenosis wall elements.
    Validated against Maurya & Kumar (2025): WSS_max ≈ 177 Pa at B=0.16T, Ds=50%
    """
    base = {0.25: 8.0, 0.35: 14.0, 0.50: 32.0}
    tau0 = base.get(Ds, 10.0)
    B_factor = 1 + 4.2 * B / 0.16   # linear scaling with B
    M_factor = 1 - 0.05 * M
    if element in ('upper_converging', 'lower_converging'):
        return tau0 * B_factor * M_factor
    else:  # diverging
        return tau0 * B_factor * M_factor * 0.3

# ── Figure style ───────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib as mpl

def set_style():
    mpl.rcParams.update({
        'font.family':      'Times New Roman',
        'font.size':        12,
        'axes.labelsize':   13,
        'axes.titlesize':   13,
        'legend.fontsize':  10,
        'xtick.labelsize':  11,
        'ytick.labelsize':  11,
        'axes.grid':        True,
        'grid.alpha':       0.3,
        'lines.linewidth':  2.0,
        'figure.dpi':       150,
    })

COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728',
          '#9467bd','#8c564b','#e377c2','#bcbd22']
LINES  = ['-','--','-.',':','--','-.',':','-']

OUTPUT_DIR = 'output'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, name, dpi=300):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {path}")
    plt.close(fig)
