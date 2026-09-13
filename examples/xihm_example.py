"""Quick look at the bundled halo-matter correlation function.

``XiHMEmulator`` composes four things: an emulated bias *ratio*, a linear matter
correlation function from the bundled P(k) via FFTLog, a linear-bias baseline
that takes over at large separations, and a blend between the two. The first
panel below shows the emulated ratio -- flat in r by construction, since the
scale dependence lives in ``xi_mm`` -- and the second the composed answer, where
the turnover past ~40 Mpc/h is the baseline taking over.

Run with ``MPLBACKEND=Agg`` on a machine without a display. Uses matplotlib,
which is not a dependency of the package.
"""

import matplotlib.pyplot as plt
import numpy as np

from jet.emulator.xihm import LGDEN_GRID, XiHMEmulator

emu = XiHMEmulator.load()
theta = np.array([0.049, 0.31, 67.66, 0.9665, 2.1e-9, -1.0, 0.0, 0.06])

# Out to 500 Mpc/h, well past the emulated box's 42.5. That is the realistic
# case, and it is why the blend exists: past ~80 the answer is the linear-bias
# baseline and the extrapolated part carries no weight. Warning suppressed
# because the extrapolation is understood here, not because it is invisible.
emu.warn_on_extrapolation = False
r = np.logspace(-1.5, 2.7, 200)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Left: the raw emulated quantity, on the box's own axes.
ax = axes[0]
for lgden in LGDEN_GRID:
    ax.loglog(r, emu.brhm(theta, z=0.0, lgden=lgden, r=r)[0, 0, 0, :], label=f"{lgden:.1f}")
ax.set(xlabel="r  [Mpc/h]", ylabel="xi_hm / xi_mm", title="emulated ratio, z = 0")
ax.legend(title=r"log10 n(>=M)", fontsize=7, title_fontsize=7)
ax.grid(True, which="both", alpha=0.3)

# Middle: the composed correlation function, with the blend's switch marked.
ax = axes[1]
for lgden in LGDEN_GRID[::2]:
    ax.loglog(
        r,
        emu.xihm_lgnbar_threshold(theta, z=0.0, r=r, lgden=lgden)[0, 0, 0, :],
        label=f"{lgden:.1f}",
    )
ax.axvline(40.0, color="gray", ls=":", lw=1, label="blend switch")
ax.set(xlabel="r  [Mpc/h]", ylabel="xi_hm", title="correlation function, z = 0")
ax.legend(title=r"log10 n(>=M)", fontsize=7, title_fontsize=7)
ax.grid(True, which="both", alpha=0.3)

# Right: the same at a fixed threshold, across redshift, against xi_mm.
ax = axes[2]
for z in (0.0, 1.0, 3.0):
    ax.loglog(
        r, emu.xihm_lgnbar_threshold(theta, z=z, r=r, lgden=-4.0)[0, 0, 0, :], label=f"z = {z}"
    )
ax.loglog(r, emu.ximm(theta, z=0.0, r=r)[0, 0, :], "k--", lw=1, label="xi_mm, z = 0")
ax.set(xlabel="r  [Mpc/h]", ylabel="xi", title=r"log10 n(>=M) = -4.0")
ax.legend(fontsize=7)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
