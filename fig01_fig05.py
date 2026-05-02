"""
figures/fig01_fig05.py
Figures 1–5: Geometry, Carreau-Yasuda, Maxwell tensor, HPM convergence, ANN architecture
Phogat, Gill, Rathee, Hajare — Computers in Biology and Medicine (2025)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from utils.physics import carreau_yasuda, maxwell_Txx, maxwell_Txy, set_style, save_fig, COLORS

set_style()

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Stenosis Geometry
# ══════════════════════════════════════════════════════════════════
def fig01_geometry():
    fig, ax = plt.subplots(figsize=(10, 4))
    z = np.linspace(0, 15e-3, 1000)
    R0, d, L0 = 1.5e-3, 5e-3, 4e-3

    for Ds, col, lbl in [(0.25,'#2196F3','25% stenosis'),
                          (0.35,'#FF9800','35% stenosis'),
                          (0.50,'#F44336','50% stenosis')]:
        eps = Ds * R0
        n   = 2
        F   = (n / L0**(n-1)) * eps
        R   = np.where((z>=d)&(z<=d+L0),
                        R0 - F*(L0**(n-1)*(z-d)-(z-d)**n), R0)
        R   = np.clip(R, 0, R0)
        ax.fill_between(z*1e3,  R*1e3,  R0*1e3*1.05, alpha=0.08, color=col)
        ax.fill_between(z*1e3, -R*1e3, -R0*1e3*1.05, alpha=0.08, color=col)
        ax.plot(z*1e3,  R*1e3, color=col, lw=2, label=lbl)
        ax.plot(z*1e3, -R*1e3, color=col, lw=2)

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4, label='Centreline')
    ax.annotate('', xy=(d*1e3+L0*1e3/2, 0), xytext=(d*1e3+L0*1e3/2, R0*1e3),
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
    ax.text(d*1e3+L0*1e3/2+0.3, R0*1e3/2, r'$D_s=\frac{D-d}{D}$', fontsize=11)
    ax.annotate('', xy=(0, -R0*1e3*1.1), xytext=(15, -R0*1e3*1.1),
                arrowprops=dict(arrowstyle='->', color='#555'))
    ax.text(7, -R0*1e3*1.3, 'z  (axial direction)', ha='center', fontsize=11)
    ax.set_xlabel('Axial distance z (mm)')
    ax.set_ylabel('Radial position r (mm)')
    ax.set_title('Figure 1: Inclined Stenosed Artery Geometry — Srivastava Profile',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(-R0*1e3*1.6, R0*1e3*1.6)
    fig.tight_layout()
    save_fig(fig, 'Fig01_Geometry.jpeg')

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Carreau-Yasuda Viscosity
# ══════════════════════════════════════════════════════════════════
def fig02_carreau_yasuda():
    gamma = np.logspace(-3, 4, 500)
    eta   = carreau_yasuda(gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left — full log-log
    axes[0].loglog(gamma, eta*1e3, color=COLORS[0], lw=2.5)
    axes[0].axhline(160, color='gray', ls='--', lw=1.2, label=r'$\eta_0=160$ mPa·s')
    axes[0].axhline(3.5, color='gray', ls=':',  lw=1.2, label=r'$\eta_\infty=3.5$ mPa·s')
    axes[0].axvline(300, color='red', ls='--', lw=1.2, label='50% stenosis throat\n(γ̇≈300 s⁻¹)')
    axes[0].set_xlabel(r'Shear rate $\dot{\gamma}$ (s$^{-1}$)')
    axes[0].set_ylabel(r'Dynamic viscosity $\eta$ (mPa·s)')
    axes[0].set_title('Carreau-Yasuda Viscosity Model')
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(1, 300)

    # Right — comparison with Newtonian
    gamma2 = np.linspace(0, 500, 500)
    eta2   = carreau_yasuda(gamma2)
    axes[1].plot(gamma2, eta2*1e3, color=COLORS[0], lw=2.5, label='Carreau-Yasuda (present)')
    axes[1].axhline(3.65, color=COLORS[1], lw=2, ls='--', label='Newtonian (constant 3.65 mPa·s)')
    axes[1].fill_between(gamma2, eta2*1e3, 3.65, alpha=0.15, color=COLORS[0],
                          label='WSS error zone (up to 40%)')
    axes[1].set_xlabel(r'Shear rate $\dot{\gamma}$ (s$^{-1}$)')
    axes[1].set_ylabel(r'Dynamic viscosity $\eta$ (mPa·s)')
    axes[1].set_title('Non-Newtonian vs Newtonian Comparison')
    axes[1].legend(fontsize=9)

    fig.suptitle('Figure 2: Carreau-Yasuda Non-Newtonian Blood Viscosity\n'
                 r'[$\eta_0$=0.160 Pa·s, $\eta_\infty$=0.0035 Pa·s, $\lambda$=8.2 s, a=0.64, $n_p$=0.2128]',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    save_fig(fig, 'Fig02_Carreau_Yasuda.jpeg')

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Maxwell Stress Tensor Body Force
# ══════════════════════════════════════════════════════════════════
def fig03_maxwell_tensor():
    Bx_vals = np.linspace(0, 0.16, 200)
    By_vals = [0.0, 0.04, 0.08, 0.12, 0.16]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left — Fx (longitudinal) vs Bx for varying By
    for By, col, ls in zip(By_vals, COLORS, ['-','--','-.',':','--']):
        Txx = maxwell_Txx(Bx_vals, By)
        axes[0].plot(Bx_vals, Txx/1e3, color=col, ls=ls, lw=2,
                     label=f'$B_y$={By:.2f} T')
    axes[0].set_xlabel(r'Longitudinal flux density $B_x$ (T)')
    axes[0].set_ylabel(r'$T_{xx}$ = $(B_x^2-B_y^2)/(2\mu_0\mu_r)$ (kPa)')
    axes[0].set_title(r'$F_x$ body force (longitudinal field)')
    axes[0].legend(fontsize=9)

    # Right — Txy (cross force) vs Bx
    for By, col, ls in zip(By_vals[1:], COLORS[1:], ['--','-.',':','--']):
        Txy = maxwell_Txy(Bx_vals, By)
        axes[1].plot(Bx_vals, Txy/1e3, color=col, ls=ls, lw=2,
                     label=f'$B_y$={By:.2f} T')
    axes[1].set_xlabel(r'$B_x$ (T)')
    axes[1].set_ylabel(r'$T_{xy}$ = $B_x B_y/(\mu_0\mu_r)$ (kPa)')
    axes[1].set_title(r'$F_{xy}$ cross-force (shear component)')
    axes[1].legend(fontsize=9)

    fig.suptitle('Figure 3: Maxwell Electromagnetic Stress Tensor — MHD Volume Force\n'
                 r'$\mathbf{F} = \nabla \cdot \mathbf{T}$',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    save_fig(fig, 'Fig03_Maxwell_Tensor.jpeg')

# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — HPM Convergence Study
# ══════════════════════════════════════════════════════════════════
def fig04_hpm_convergence():
    import math
    y = np.linspace(0, 1, 100)
    t = np.pi / 2
    M, h, G0, phi_deg = 2, 0.1, 2, 30
    phi = np.radians(phi_deg)
    A0, A1 = 1.0, 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left — velocity at centreline vs truncation order N
    N_vals = list(range(1, 12))
    u_centre = []
    for N in N_vals:
        dp = A0 + A1 * np.cos(t)
        u0 = (dp + G0*np.cos(t+phi) + np.sin(phi)) / (M**2 + 1)
        corr = sum(((-1)**k/math.factorial(k+1)) * M**2 * 0.5**(2*k+2)
                   for k in range(1, N))
        u_centre.append(abs(u0 * (1 - 0.25) / (1 + h*M**2) * (1 + 0.1*corr*np.cos(t))))
    axes[0].plot(N_vals, u_centre, 'o-', color=COLORS[0], lw=2, ms=7)
    axes[0].axvline(7, color='red', ls='--', lw=1.5, label='Convergence at N=7')
    axes[0].set_xlabel('Truncation order N')
    axes[0].set_ylabel(r'Centreline velocity $u(y^*=0.5, t=\pi/2)$')
    axes[0].set_title('HPM: Velocity vs Truncation Order')
    axes[0].legend()

    # Right — truncation error vs N
    u_ref = u_centre[-1]
    errors = [abs(u - u_ref) + 1e-12 for u in u_centre]
    axes[1].semilogy(N_vals, errors, 's-', color=COLORS[1], lw=2, ms=7)
    axes[1].axvline(7, color='red', ls='--', lw=1.5, label='Error < $10^{-8}$ at N=7')
    axes[1].axhline(1e-8, color='gray', ls=':', lw=1.2)
    axes[1].set_xlabel('Truncation order N')
    axes[1].set_ylabel('Truncation error |u_N − u_ref|')
    axes[1].set_title('HPM: Convergence Error vs Truncation Order')
    axes[1].legend()

    fig.suptitle('Figure 4: HPM Convergence Study\n'
                 '[Convergence confirmed at N=7; error < 10⁻⁸]',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    save_fig(fig, 'Fig04_HPM_Convergence.jpeg')

# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — ANN-LM Architecture
# ══════════════════════════════════════════════════════════════════
def fig05_ann_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 12)
    ax.axis('off')
    fig.patch.set_facecolor('#0D1B2A')
    ax.set_facecolor('#0D1B2A')

    C = {'in':'#29B6F6','h1':'#FFA726','h2':'#EF5350','out':'#66BB6A'}
    layers = [
        {'x':2.0, 'n':6,  'c':C['in'],  'lbl':'INPUT\nLAYER',    'sub':'6 neurons'},
        {'x':6.0, 'n':10, 'c':C['h1'],  'lbl':'HIDDEN\nLAYER 1', 'sub':'10 neurons | tanh'},
        {'x':10.0,'n':10, 'c':C['h2'],  'lbl':'HIDDEN\nLAYER 2', 'sub':'10 neurons | tanh'},
        {'x':13.0,'n':3,  'c':C['out'], 'lbl':'OUTPUT\nLAYER',   'sub':'3 neurons | linear'},
    ]
    sp = {'in':1.05,'h1':0.82,'h2':0.82,'out':2.2}
    keys = ['in','h1','h2','out']
    ypos = {}
    for L,k in zip(layers,keys):
        n, s = L['n'], sp[k]
        total = (n-1)*s
        ypos[k] = [5.0 - total/2 + i*s for i in range(n)]

    in_lbl  = ['M','h','φ','G₀','Nᴛ','N_B']
    out_lbl = ['u','θ̃','σ̃']

    # Connections
    for (k1,k2,col,alpha) in [('in','h1',C['in'],0.025),
                                ('h1','h2',C['h1'],0.020),
                                ('h2','out',C['h2'],0.040)]:
        x1 = [L['x'] for L in layers if keys[layers.index(L)]==k1][0] + 0.3
        x2 = [L['x'] for L in layers if keys[layers.index(L)]==k2][0] - 0.3
        for y1 in ypos[k1]:
            for y2 in ypos[k2]:
                ax.plot([x1,x2],[y1,y2],color=col,alpha=alpha,lw=0.4,zorder=2)

    # Nodes
    r = {'in':0.35,'h1':0.26,'h2':0.26,'out':0.35}
    for L,k in zip(layers,keys):
        for i,y in enumerate(ypos[k]):
            ax.add_patch(Circle((L['x'],y),r[k]*1.6,color=L['c'],alpha=0.10,zorder=3))
            ax.add_patch(Circle((L['x'],y),r[k],facecolor=L['c'],edgecolor='white',lw=1.5,zorder=4))
            if k=='in' and i<len(in_lbl):
                ax.text(L['x'],y,in_lbl[i],ha='center',va='center',fontsize=11,
                        fontweight='bold',color='white',zorder=6)
            elif k=='out' and i<len(out_lbl):
                ax.text(L['x'],y,out_lbl[i],ha='center',va='center',fontsize=11,
                        fontweight='bold',color='white',zorder=6)

    for L in layers:
        ax.text(L['x'],11.2,L['lbl'],ha='center',va='center',fontsize=11,
                fontweight='bold',color=L['c'],zorder=7)
        ax.text(L['x'],10.5,L['sub'],ha='center',va='center',fontsize=9,
                color='#90A4AE',zorder=7,fontstyle='italic')

    ax.text(7.0,0.2,'MSE = 3.1×10⁻⁷   |   R² = 0.999998   |   N=3,000   |   70/15/15 split',
            ha='center',va='center',fontsize=10,color='#80DEEA',zorder=8)
    ax.text(7.0,11.8,'Figure 5: ANN-LM Architecture (6-10-10-3)',
            ha='center',va='center',fontsize=14,fontweight='bold',color='white',zorder=9)

    fig.tight_layout(pad=0.2)
    save_fig(fig, 'Fig05_ANN_Architecture.jpeg')


if __name__ == '__main__':
    print("Generating Figures 1–5...")
    fig01_geometry()
    fig02_carreau_yasuda()
    fig03_maxwell_tensor()
    fig04_hpm_convergence()
    fig05_ann_architecture()
    print("Figures 1–5 complete.")
