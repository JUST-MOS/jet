"""Quick look at the baryonic halo mass function.

The whole point of this model is the *baryon* axis, so the plot varies one
feedback parameter at a time and holds the rest at the reference's defaults.
``logMc`` is the halo mass above which feedback suppresses the gas: raising it
pushes the suppression to more massive haloes, which is visible as a deficit
that moves right and deepens.

There is no redshift and no mass interpolation here -- the weights are one
cosmology at one redshift, on a fixed grid. See `jet.emulator.hmf_bcm`.

Run with ``MPLBACKEND=Agg`` on a machine without a display. Uses matplotlib,
which is not a dependency of the package.
"""

import matplotlib.pyplot as plt
import numpy as np

from jet.emulator.hmf_bcm import BCMHFEmulator, mass_grid, theta_spec

theta_default = np.array([15, 1.5, 3.5, 4.5])

emu = BCMHFEmulator.load()
M = mass_grid()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Left and middle: the cumulative abundance and its derivative, for the default
# point, which is where a caller starts.
ax = axes[0]
n = emu.cumulative(theta_default[None, :])[0]
ax.loglog(M, n)
ax.set(xlabel="M  [Msun/h]", ylabel="n(>=M)  [(h/Mpc)^3]", title="reference defaults")

ax = axes[1]
ax.loglog(M, emu.dndlgM(theta_default[None, :])[0])
ax.set(xlabel="M  [Msun/h]", ylabel="dn/dlog10(M)", title="differential (binned)")

# Right: the baryon axis, which is what this model is for.
ax = axes[2]
for log_mc in (14.0, 15.0, 16.0):
    theta = np.copy(theta_default)
    theta[theta_spec().index("logMc")] = log_mc
    ax.loglog(M, emu.cumulative(theta[None, :])[0] / n, label=f"logMc = {log_mc:.0f}")
ax.set(xlabel="M  [Msun/h]", ylabel="n(>=M) ratio", title="varying the feedback scale")
ax.legend(fontsize=7)

for ax in axes:
    ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
