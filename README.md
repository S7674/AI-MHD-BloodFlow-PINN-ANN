# MHD Blood Flow — Figure Generation Code
## AI-Augmented Multi-Method Analysis of MHD Heat and Mass Transfer in Blood-Based Nanofluid through an Inclined Porous Stenosed Artery

**Authors:** Neha Phogat, Sumeet Gill, Rajbala Rathee, Sunil Tulshiram Hajare*  
**Journal:** Computers in Biology and Medicine (Elsevier)  
**Corresponding author:** sunilhajare@gmail.com

---

## Repository Structure

```
mhd_bloodflow_figures/
├── README.md
├── requirements.txt
├── run_all.py               # Run all 40 figures at once
├── utils/
│   └── physics.py           # Shared physics functions (HPM, AGM, Carreau-Yasuda)
├── figures/
│   ├── fig01_geometry.py
│   ├── fig02_carreau_yasuda.py
│   ├── fig03_maxwell_tensor.py
│   ├── fig04_hpm_convergence.py
│   ├── fig05_ann_architecture.py
│   ├── fig06_velocity_Bx.py
│   ├── fig07_velocity_comparison.py
│   ├── fig08_velocity_By.py
│   ├── fig09_radial_velocity.py
│   ├── fig10_pressure_variation.py
│   ├── fig11_stenosis_shapes.py
│   ├── fig12_wss_no_field.py
│   ├── fig13_wss_Bx.py
│   ├── fig14_wss_By.py
│   ├── fig15_time_dependent.py
│   ├── fig16_velocity_vs_M.py
│   ├── fig17_velocity_vs_h.py
│   ├── fig18_velocity_vs_G0.py
│   ├── fig19_velocity_vs_phi.py
│   ├── fig20_flowrate_vs_M.py
│   ├── fig21_flowrate_vs_h.py
│   ├── fig22_flowrate_vs_G0.py
│   ├── fig23_flowrate_vs_phi.py
│   ├── fig24_lipid_threshold.py
│   ├── fig25_entropy_generation.py
│   ├── fig26_bejan_number.py
│   ├── fig27_pinn_validation.py
│   ├── fig28_wss_vs_h.py
│   ├── fig29_temperature_NT.py
│   ├── fig30_concentration_NB.py
│   ├── fig31_ann_mse.py
│   ├── fig32_ann_regression.py
│   ├── fig33_error_histogram.py
│   ├── fig34_training_state.py
│   ├── fig35_actual_vs_predicted.py
│   ├── fig36_rsm_M_h.py
│   ├── fig37_rsm_G0_phi.py
│   ├── fig38_sensitivity.py
│   ├── fig39_ann_regression_theta_sigma.py
│   └── fig40_hpm_vs_ann.py
```

## Physical Parameters

| Parameter | Value | Description |
|---|---|---|
| η₀ | 0.160 Pa·s | Zero shear-rate viscosity |
| η∞ | 0.0035 Pa·s | Infinite shear-rate viscosity |
| λ | 8.2 s | Relaxation time |
| a | 0.64 | Yasuda parameter |
| nₚ | 0.2128 | Flow behaviour index |
| ρ | 1060 kg/m³ | Blood density |
| Uav | 0.5 m/s | Average inlet velocity |
| B_max | 0.16 T | Maximum magnetic flux density |

## Installation

```bash
pip install numpy matplotlib scipy scikit-learn
```

## Usage

Run individual figures:
```bash
python figures/fig02_carreau_yasuda.py
```

Run all figures:
```bash
python run_all.py
```

Figures are saved as high-resolution JPEG (300 dpi) in `output/` directory.

## Citation

If you use this code, please cite:
> Phogat N., Gill S., Rathee R., Hajare S.T. (2025). AI-Augmented Multi-Method Analysis of MHD Heat and Mass Transfer in Blood-Based Nanofluid through an Inclined Porous Stenosed Artery. *Computers in Biology and Medicine*, Elsevier.

## License
MIT License — free to use with attribution.
