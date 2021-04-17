import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # Parse cmd-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--noplot", action='store_true')
    parser.add_argument("-f", "--filename")
    parser.add_argument("-e", "--errormag")
    args = parser.parse_args()
    noplot = args.noplot
    filename = args.filename if args.filename else "2pt_hisq_msml5_fine_D_nongold_489conf"
    errormag = int(args.errormag if args.errormag else 1)
    # Load data
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
    
    if not noplot:
        plot(meff, avgmeff, f"Effective mass for n={len(meff)} gauge configurations\non a fine lattice", f"meff_{filename}_title", errormag)
        plot(meff, avgmeff, None, f"meff_{filename}", errormag)


def plot(meff_data, avg_meff, title, filename, errormag):
    avg = meff_data.mean(0)
    stddev = meff_data.std(0)
    time_slices = len(avg)

    errormaglabel = "" if errormag==1 else f" (Errors magnified by x{errormag})"

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_title(title, fontsize=24)

    ax.errorbar(range(time_slices), avg, yerr=errormag*stddev, ecolor="black", elinewidth=1, linewidth=0, fmt='r.', markersize=2, \
            label=r"$m_{eff}(t)$" + errormaglabel)
    ax.axhline(avg_meff, color='blue', linestyle=':', label=r"First order estimate for $m_{eff} \approx$ " + str(round(avg_meff, 2)) )

    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Mass", fontsize=18)
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='both', labelsize=16)

    fig.tight_layout()
    plt.savefig(f"images/{filename}.png")

main()
