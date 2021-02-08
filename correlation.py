import numpy as np
import matplotlib.pyplot as plt
import sys
from plot import plot

if len(sys.argv) == 1:
	filename = "2pt_hisq_msml5_fine_D_nongold_489conf"
else:
	filename = sys.argv[1]

data = np.loadtxt(f"data/{filename}.dat")
plot(data, title=f"Average 2-point correlator for n={len(data)} gauge configurations\non a fine lattice", ylabel="Correlation", filename=f"correlation_{filename}")

