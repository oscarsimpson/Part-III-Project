import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/jacknife.dat")

avg = data.mean(0)
stddev = data.std(0)
time_slices = len(avg)

avg = np.append(avg, avg[0])
stddev = np.append(stddev, stddev[0])

fig, ax = plt.subplots(figsize=(12, 7.2))
ax.set_title("Average 2-point correlator for n=489 gauge configurations\non a fine lattice", fontsize=24)
ax.set_xticks(np.linspace(0, time_slices, 13))

ax.plot(range(time_slices+1), avg, 'b+', linewidth=0)
ax.set_xlabel("Time", fontsize=18)
ax.set_ylabel("Correlation", fontsize=18)
ax.tick_params(axis='both', which='both', labelsize=16)

ax2 = ax.twinx()
ax2.plot(stddev, color='orange', linewidth=0.8)
ax2.set_yscale('log')
ax2.set_ylabel('Error', fontsize=18)
ax2.tick_params(axis='both', which='both', labelsize=16)

fig.tight_layout()
plt.savefig("correlator.png")
