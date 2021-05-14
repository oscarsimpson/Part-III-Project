import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # Parse cmd-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--noplot", action='store_true')
    parser.add_argument("-f", "--filename")
    parser.add_argument("-e", "--errormag")
    parser.add_argument("-s", "--platstart")
    args = parser.parse_args()
    noplot = args.noplot
    filename = args.filename if args.filename else "2pt_hisq_msml5_fine_D_nongold_489conf"
    errormag = int(args.errormag if args.errormag else 1)
    plat_start = int(args.platstart if args.platstart else 24)
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
    # This should be checked for each dataset. i.e. D_nongold is ~24-36, D_gold is ~16-36, K_zeromom is ~8-36
    # It is very important to use an even number for the range, as there is some oscillatory behaviour. 
    # By using an even range, we make sure that some of this is cancelled out between the first and the last data point.
    # If an odd range is used, the oscillations compound.
    # This is why we do not use plat_last+1 in the slice.
    # plat_start = 8
    plat_last = 36
    avgmeff = (meff.T)[plat_start:plat_last].mean()
    print(f"First estimate for m_eff = {avgmeff}")
    
    if not noplot:
        # plot(meff, avgmeff, f"Effective mass for n={len(meff)} gauge configurations\non a fine lattice", f"meff_{filename}_title", errormag)
        plot(meff, avgmeff, None, f"meff_{filename}", errormag, (plat_start, plat_last))


def plot(meff_data, avg_meff, title, filename, errormag, plateau):
    avg = meff_data.mean(0)
    stddev = meff_data.std(0)
    time_slices = len(avg)

    errormaglabel = "" if errormag==1 else f" (Errors magnified by x{errormag})"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title, fontsize=24)

    ax.errorbar(range(time_slices), avg, yerr=errormag*stddev, ecolor="black", elinewidth=1.5, linewidth=0, fmt='r.', markersize=6, \
            label=r"$m_{eff}(t)$" + errormaglabel)
    ax.axhline(avg_meff, color='blue', linestyle=':', label=r"First order estimate for $m_{eff} \approx$ " + str(round(avg_meff, 3)) )
    ax.axvline(x=plateau[0]-0.5, color='green', linestyle=':')
    ax.axvline(x=plateau[1]-0.5, color='green', linestyle=':')
    # The -0.5 makes it very clear which points are included in the average.

    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Effective Mass", fontsize=18)
    ax.legend(fontsize=16, loc='upper right')
    ax.tick_params(axis='both', which='both', labelsize=16)

    fig.tight_layout()
    plt.savefig(f"images/{filename}.png")

main()
