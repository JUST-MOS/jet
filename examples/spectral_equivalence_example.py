"""Basic usage of the cosmological spectral-equivalence module.

The example has four parts:

1. distance matching for a CPL cosmology whose dark-energy parameters are
   outside the P(k) emulator's training box -- distance matching does not use
   the emulator, so this is allowed;
2. amplitude matching for a cosmology inside the training box, using the
   bundled total-matter P(k) emulator;
3. the same call on an out-of-box cosmology, which now raises ``ValueError``
   instead of silently extrapolating;
4. the optional CAMB fallback for auxiliary models that leave the box.

Run with::

    python examples/spectral_equivalence_example.py

The bundled Pcb and Pnn/Pcb weights must be present for the emulator paths.
They are distributed separately from git; see ``jet/data/README.md``.
"""

import numpy as np

from jet.emulator.pklin import theta_spec
from jet.equivalence import spectral_equivalence_as, spectral_equivalence_de

# theta columns:
#   Omegab, Omegam, H0, ns, As, w0, wa, mnu
# Omegam is baryons + cold dark matter, excluding massive neutrinos. As is the
# physical amplitude, not As * 1e9.
theta_reference = np.array(
    [0.055, 0.35140742, 63.6, 0.964, 2.082644720532909e-09, -0.42, -1.75, 0.06]
)
z_reference = np.array([0.0, 1.0, 2.0, 3.0])

print("=" * 72)
print("1. Distance matching (wCDM and w0waCDM)")
print("=" * 72)

# The original w0/wa are outside the emulator box, but distances only need the
# background expansion, so no emulator is involved here.
w_const, err_w = spectral_equivalence_de(theta_reference, z_reference, wcdm=True, return_error=True)
print("auxiliary wCDM w(z):")
for zz, value, error in zip(z_reference, w_const[0], err_w[0]):
    print(f"  z = {zz:4.2f}   w = {value: .8f}   |dchi|/chi = {error:.3e}")

w0_eq, wa_eq, err_chi = spectral_equivalence_de(theta_reference, z_reference, return_error=True)
print("\nauxiliary w0waCDM model:")
for zz, w0_value, wa_value, error in zip(z_reference, w0_eq[0], wa_eq[0], err_chi[0]):
    print(
        f"  z = {zz:4.2f}   w0 = {w0_value: .8f}   wa = {wa_value: .8f}   |dchi|/chi = {error:.3e}"
    )

# A cosmology inside the P(k) emulator's training box. The bounds are checked
# explicitly here so the example states its assumption rather than implying it.
theta_in_box = np.array([0.05, 0.31, 68.0, 0.965, 2.1e-09, -1.1, 0.2, 0.06])
z_in_box = np.array([0.5, 1.0, 2.0])
theta_spec().require_within_bounds(theta_in_box, name="theta_in_box")

print()
print("=" * 72)
print("2. Amplitude matching with the bundled emulator")
print("=" * 72)

w_const_in_box = spectral_equivalence_de(theta_in_box, z_in_box, wcdm=True)
As_eq = spectral_equivalence_as(theta_in_box, z_in_box, w_const_in_box, sigma8_type="emulator")
print("auxiliary wCDM w(z):", w_const_in_box[0])
print("equivalent As(z):   ", As_eq[0])
print("original As:        ", theta_in_box[4])

print()
print("=" * 72)
print("3. Out-of-box input with sigma8_type='emulator'")
print("=" * 72)

try:
    spectral_equivalence_as(theta_reference, z_reference, w0_eq, wa_eq, sigma8_type="emulator")
except ValueError as exc:
    print("raised ValueError as expected:")
    print(exc)

# A cosmology whose *auxiliary* wCDM model also leaves the box. This is the
# case where CAMB can rescue the amplitude matching if the caller opts in.
theta_out_of_box = np.array([0.05, 0.32, 70.0, 0.96, 2.1e-09, -1.25, -0.5, 0.0])
z_out_of_box = np.array([0.5, 1.0, 2.0])
w_const_out_of_box = spectral_equivalence_de(theta_out_of_box, z_out_of_box, wcdm=True)

print()
print("=" * 72)
print("4. Auxiliary model outside the box: CAMB fallback")
print("=" * 72)

try:
    spectral_equivalence_as(
        theta_out_of_box,
        z_out_of_box,
        w_const_out_of_box,
        sigma8_type="camb",
    )
except ValueError as exc:
    print("default (fallback_to_camb=False) raises:")
    print(exc)

try:
    As_eq_fallback = spectral_equivalence_as(
        theta_out_of_box,
        z_out_of_box,
        w_const_out_of_box,
        sigma8_type="camb",
        fallback_to_camb=True,
    )
except ImportError as exc:
    print(f"\nCAMB fallback skipped: {exc}")
else:
    print("\nfallback_to_camb=True gives:")
    print("  auxiliary wCDM w(z):", w_const_out_of_box[0])
    print("  equivalent As(z):   ", As_eq_fallback[0])
    print("  original As:        ", theta_out_of_box[4])
