"""Basic usage of the cosmological spectral-equivalence module.

The example has three parts:

1. distance matching for a CPL cosmology whose dark-energy parameters are
   outside the emulator training box -- the case the reference notebooks use;
2. amplitude matching for a cosmology inside the training box, using the
   bundled P(k) emulator;
3. the same amplitude matching with CLASS, if ``classy`` is installed.

Run with::

    python examples/equivalence_example.py

The bundled P(k) weights must be present. They are distributed separately from
git; see ``jet/data/README.md``.
"""

import numpy as np

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

w_const = spectral_equivalence_de(theta_reference, z_reference, wcdm=True)
print("auxiliary wCDM w(z):")
for zz, value in zip(z_reference, w_const[0]):
    print(f"  z = {zz:4.2f}   w = {value: .8f}")

w0_eq, wa_eq, err_chi = spectral_equivalence_de(theta_reference, z_reference, return_error=True)
print("\nauxiliary w0waCDM model:")
for zz, w0_value, wa_value, error in zip(z_reference, w0_eq[0], wa_eq[0], err_chi[0]):
    print(
        f"  z = {zz:4.2f}   w0 = {w0_value: .8f}   wa = {wa_value: .8f}   |dchi|/chi = {error:.3e}"
    )

# The reference model's w0/wa lie outside the emulator training box, so the
# emulator cannot supply its target sigma8. CLASS or CAMB is the right backend
# for that cosmology. The in-box example below uses the emulator instead.
theta_in_box = np.array([0.05, 0.31, 68.0, 0.965, 2.1e-09, -1.1, 0.2, 0.06])
z_in_box = np.array([0.5, 1.0, 2.0])

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
print("3. Amplitude matching with CAMB (optional)")
print("=" * 72)

try:
    As_eq_class = spectral_equivalence_as(
        theta_reference, z_reference, w0_eq, wa_eq, sigma8_type="camb"
    )
except ImportError as exc:
    print(f"skipped: {exc}")
else:
    print("equivalent As(z) from CAMB:", As_eq_class[0])
    print("original As:                ", theta_reference[4])
