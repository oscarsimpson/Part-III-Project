import numpy as np
from plot import plot
import sys

if len(sys.argv) == 1:
	filename = "2pt_hisq_msml5_fine_D_nongold_489conf"
else:
	filename = sys.argv[1]

data = np.loadtxt(f"data/{filename}.dat")
num_confs = len(data)
# We go halfway through because after this point the correlation begins to increase. This leads to a negative effective mass.
time_slices = len(data[0])//2

meff = np.empty_like(data)
for i in range(num_confs):
	for j in range(time_slices):
		meff[i, j] = np.log(np.abs(data[i, j] / data[i, j+1]))
meff = ((meff.T)[:time_slices]).T

# We expect the m_eff values to have plateaud about 1/2 - 3/4 of the way to the halfway point.
# Visual inspection of m_eff(t) supports this.

avgmeff = (meff.T)[time_slices//2:3*time_slices//4].mean()
print(f"First estimate for m_eff = {avgmeff}")

plot(meff, title=f"Effective mass for n={len(meff)} gauge configurations\non a fine lattice", ylabel=r"$m_{eff}(t)$", filename=f"meff_{filename}", avg_data=avgmeff)


