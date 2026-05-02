"""
run_all.py
Master script — generates all 40 figures for:
"AI-Augmented Multi-Method Analysis of MHD Heat and Mass Transfer in
Blood-Based Nanofluid through an Inclined Porous Stenosed Artery"
Phogat, Gill, Rathee, Hajare — 

Usage:
    python run_all.py

Output: all figures saved to ./output/ at 300 dpi
"""
import os, sys, time

os.makedirs('output', exist_ok=True)
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 65)
print("  MHD Blood Flow Figure Generator")
print("  Phogat, Gill, Rathee, Hajare — CBM (2025)")
print("=" * 65)

from figures.fig01_fig05 import (fig01_geometry, fig02_carreau_yasuda,
                                   fig03_maxwell_tensor, fig04_hpm_convergence,
                                   fig05_ann_architecture)
from figures.fig06_fig15 import (fig06_velocity_Bx, fig07_velocity_comparison,
                                   fig08_velocity_By, fig09_radial_velocity,
                                   fig10_pressure, fig11_stenosis_shapes,
                                   fig12_wss_no_field, fig13_wss_Bx,
                                   fig14_wss_By, fig15_time_dependent)
from figures.fig16_fig30 import (fig16_velocity_M, fig17_velocity_h,
                                   fig18_velocity_G0, fig19_velocity_phi,
                                   fig20_Q_M, fig21_Q_h, fig22_Q_G0, fig23_Q_phi,
                                   fig24_lipid_threshold, fig25_entropy,
                                   fig26_bejan, fig27_pinn_validation,
                                   fig28_wss_h, fig29_temperature_NT,
                                   fig30_concentration_NB)
from figures.fig31_fig40 import (fig31_mse, fig32_regression, fig33_histogram,
                                   fig34_training_state, fig35_actual_predicted,
                                   fig36_rsm_M_h, fig37_rsm_G0_phi,
                                   fig38_sensitivity, fig39_regression_theta_sigma,
                                   fig40_hpm_vs_ann)

FIGURES = [
    ("Fig 1:  Stenosis geometry",               fig01_geometry),
    ("Fig 2:  Carreau-Yasuda viscosity",         fig02_carreau_yasuda),
    ("Fig 3:  Maxwell stress tensor",            fig03_maxwell_tensor),
    ("Fig 4:  HPM convergence",                  fig04_hpm_convergence),
    ("Fig 5:  ANN-LM architecture",              fig05_ann_architecture),
    ("Fig 6:  Velocity vs Bx (4-panel)",         fig06_velocity_Bx),
    ("Fig 7:  Velocity comparison",              fig07_velocity_comparison),
    ("Fig 8:  Velocity vs By",                   fig08_velocity_By),
    ("Fig 9:  Radial velocity",                  fig09_radial_velocity),
    ("Fig 10: Pressure variation",               fig10_pressure),
    ("Fig 11: Stenosis shapes",                  fig11_stenosis_shapes),
    ("Fig 12: WSS — no field",                   fig12_wss_no_field),
    ("Fig 13: WSS under Bx",                     fig13_wss_Bx),
    ("Fig 14: WSS under By",                     fig14_wss_By),
    ("Fig 15: Time-dependent velocity",          fig15_time_dependent),
    ("Fig 16: Velocity vs M",                    fig16_velocity_M),
    ("Fig 17: Velocity vs h",                    fig17_velocity_h),
    ("Fig 18: Velocity vs G0",                   fig18_velocity_G0),
    ("Fig 19: Velocity vs phi",                  fig19_velocity_phi),
    ("Fig 20: Flow rate vs M",                   fig20_Q_M),
    ("Fig 21: Flow rate vs h",                   fig21_Q_h),
    ("Fig 22: Flow rate vs G0",                  fig22_Q_G0),
    ("Fig 23: Flow rate vs phi",                 fig23_Q_phi),
    ("Fig 24: Lipid threshold (NOVEL)",          fig24_lipid_threshold),
    ("Fig 25: Entropy generation",               fig25_entropy),
    ("Fig 26: Bejan number",                     fig26_bejan),
    ("Fig 27: PINN validation",                  fig27_pinn_validation),
    ("Fig 28: WSS vs h",                         fig28_wss_h),
    ("Fig 29: Temperature vs NT",                fig29_temperature_NT),
    ("Fig 30: Concentration vs NB",              fig30_concentration_NB),
    ("Fig 31: ANN MSE performance",              fig31_mse),
    ("Fig 32: ANN 4-panel regression",           fig32_regression),
    ("Fig 33: Error histogram",                  fig33_histogram),
    ("Fig 34: Training state",                   fig34_training_state),
    ("Fig 35: Actual vs predicted",              fig35_actual_predicted),
    ("Fig 36: RSM — M and h",                    fig36_rsm_M_h),
    ("Fig 37: RSM — G0 and phi",                 fig37_rsm_G0_phi),
    ("Fig 38: Sensitivity indices",              fig38_sensitivity),
    ("Fig 39: Regression theta sigma",           fig39_regression_theta_sigma),
    ("Fig 40: HPM vs ANN comparison",            fig40_hpm_vs_ann),
]

t0 = time.time()
for i, (name, fn) in enumerate(FIGURES, 1):
    print(f"  [{i:02d}/40] {name} ...", end=' ', flush=True)
    t1 = time.time()
    fn()
    print(f"({time.time()-t1:.1f}s)")

print()
print(f"All 40 figures generated in {time.time()-t0:.1f}s")
print(f"Output directory: ./output/")
print("=" * 65)
