"""Basic usage of the cosmological spectral-equivalence module.

The example has six parts:

1. distance matching for a CPL cosmology whose dark-energy parameters are
   outside the P(k) emulator's training box -- distance matching does not use
   the emulator, so this is allowed;
2. amplitude matching for a cosmology inside the training box, using the
   bundled total-matter P(k) emulator;
3. an out-of-box target, which raises with ``fallback_to_camb=False`` and
   falls back to CAMB on the target side with the default
   ``fallback_to_camb=True``;
4. an out-of-box auxiliary model, which falls back only on the auxiliary side;
5. the high-level ``EquivalentCosmology`` facade, which turns a sigma8-base
   theta into a per-redshift auxiliary theta and evaluates HMF/BHM/XiHM;
6. an auxiliary ``As_eq`` that leaves the box: the facade returns the theta,
   and HMF is evaluated with its own CAMB fallback.

Run with::

    python examples/spectral_equivalence_example.py

The bundled Pcb and Pnn/Pcb weights must be present for the emulator paths.
They are distributed separately from git; see ``jet/data/README.md``.
"""

import warnings

import numpy as np

from jet.emulator.hmf import HMFEmulator
from jet.emulator.pklin import theta_spec
from jet.equivalence import EquivalentCosmology, spectral_equivalence_as, spectral_equivalence_de

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
print("3. Out-of-box target: emulator with per-side CAMB fallback")
print("=" * 72)

# The target is outside the box. With fallback disabled this is an error.
try:
    spectral_equivalence_as(
        theta_reference,
        z_reference,
        w0_eq,
        wa_eq,
        sigma8_type="emulator",
        fallback_to_camb=False,
    )
except ValueError as exc:
    print("fallback_to_camb=False raises:")
    print(exc)

try:
    As_eq_default = spectral_equivalence_as(theta_reference, z_reference, w0_eq, wa_eq)
except ImportError as exc:
    print(f"\nCAMB fallback skipped: {exc}")
else:
    print("\ndefault (fallback_to_camb=True) gives:")
    print("  equivalent As(z):", As_eq_default[0])

# A cosmology whose *auxiliary* wCDM model leaves the box, while the original
# model is inside it. The default fallback switches only the auxiliary side.
theta_out_of_box = np.array([0.05, 0.32, 70.0, 0.96, 2.1e-09, -1.25, -0.5, 0.0])
z_out_of_box = np.array([0.5, 1.0, 2.0])
w_const_out_of_box = spectral_equivalence_de(theta_out_of_box, z_out_of_box, wcdm=True)

print()
print("=" * 72)
print("4. Out-of-box auxiliary: per-side CAMB fallback")
print("=" * 72)

try:
    spectral_equivalence_as(
        theta_out_of_box,
        z_out_of_box,
        w_const_out_of_box,
        sigma8_type="emulator",
        fallback_to_camb=False,
    )
except ValueError as exc:
    print("fallback_to_camb=False raises:")
    print(exc)

try:
    As_eq_fallback = spectral_equivalence_as(
        theta_out_of_box,
        z_out_of_box,
        w_const_out_of_box,
        sigma8_type="emulator",
        fallback_to_camb=True,
    )
except ImportError as exc:
    print(f"\nCAMB fallback skipped: {exc}")
else:
    print("\nfallback_to_camb=True gives:")
    print("  auxiliary wCDM w(z):", w_const_out_of_box[0])
    print("  equivalent As(z):   ", As_eq_fallback[0])
    print("  original As:        ", theta_out_of_box[4])

print()
print("=" * 72)
print("5. High-level facade: sigma8 input to an in-box auxiliary model (DESI w0-wa case)")
print("=" * 72)

# The fifth column is sigma8 at z=0 when use_sigma8=True. The facade converts
# it to As, finds the per-redshift auxiliary w0/wa, and assembles the theta.
theta_sigma8 = np.array([0.055, 0.35140742, 63.6, 0.964, 0.781, -0.42, -1.75, 0.06])
z_facade = np.array([0.0, 0.5, 1.0])
mass = np.array([1e13, 1e14])

cosmo = EquivalentCosmology(theta_sigma8, use_sigma8=True)
theta_eq = cosmo.theta(z_facade)
print("auxiliary theta (As base):")
for zz, row in zip(z_facade, theta_eq):
    print(f"  z = {zz:4.2f}   w0 = {row[5]: .8f}   wa = {row[6]: .8f}   As = {row[4]:.6e}")

# The facade's direct statistics silently use CAMB for the target sigma8; the
# expected fallback warnings are suppressed. The auxiliary model stays in box.
number_density = cosmo.number_density(z=z_facade, M=mass)
print("n(>=M) shape:", number_density.shape)

print()
print("=" * 72)
print("6. Auxiliary As outside the box: use HMF's own CAMB fallback")
print("=" * 72)

# At sigma8=0.8544 the input As is still in box, but the equivalent As_eq is
# slightly outside it. The facade returns theta_eq; only the direct statistics
# are strict and raise.
theta_edge = np.array([0.055, 0.35140742, 63.6, 0.964, 0.8544, -0.42, -1.75, 0.06])
cosmo_edge = EquivalentCosmology(theta_edge, use_sigma8=True)
theta_eq_edge = cosmo_edge.theta_at(0.0)
print("auxiliary theta (As base):", theta_eq_edge)
print("input As:                  ", cosmo_edge.theta_as[4])
print("auxiliary As_eq:           ", theta_eq_edge[4])
print("As box:                     [1.7e-09, 2.5e-09]")

try:
    cosmo_edge.number_density(z=0.0, M=mass)
except ValueError as exc:
    print("\nfacade statistics raise as expected:")
    print("  " + str(exc).splitlines()[0])

# HMF is the one statistic with a CAMB fallback; evaluate it on theta_eq.
hmf = HMFEmulator.load(fallback_to_camb=True)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    number_density_edge = hmf.number_density(theta_eq_edge, z=0.0, M=mass)
print("\nHMF fallback n(>=M) shape:", number_density_edge.shape)
print("HMF fallback warning:", caught[0].message if caught else "none")
