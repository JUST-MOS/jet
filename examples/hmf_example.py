import matplotlib.pyplot as plt
import numpy as np

from jet.emulator.hmf import (
    HMFEmulator,
    data_slices,
    mass_edges,
    mass_slices,
)

hmf = HMFEmulator.load()
theta = np.array([0.049, 0.31, 67.66, 0.9665, 2.1e-9, -1.0, 0.0, 0.06])
M_grid = np.logspace(10, 16.0, 120)
n_at_z0 = hmf.number_density(theta, z=0.0, M=M_grid)[0, 0]
blocks, bins, edges = data_slices(), mass_slices(), mass_edges()


fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(M_grid, n_at_z0)
ax.set(xlabel="M  [Msun/h]", ylabel="n(>=M)  [(h/Mpc)^3]", title="z = 0")
ylims = [1e-14, 4e-2]
plt.ylim(ylims)
plt.xlim([9e10, 2e16])
low, high = bins[0]  # index 0 corresponds to z = 0.0
ax.fill_betweenx(
    y=ylims, x1=edges[low], x2=edges[high], color="gray", alpha=0.3, label="training range"
)
plt.grid(True)
plt.show()
