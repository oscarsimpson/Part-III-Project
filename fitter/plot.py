import matplotlib.pyplot as plt
import gvar as gv 
import numpy as np
import re

import os

N_EXCL = 4 # Number of terms that we expect to be fit poorly, and will print their fitting values.
GOOD_VALS = {
        3:4,
        4:4,
        5:4,
        6:4,
        7:4,
        9:4} # for tmin = key, the n = value that we decide is a good enough fit to report a converged parameter
SPACE_VALS = {
        "coarse":0.12404,
        "fine":0.09023,
        "very-coarse":0.1509
        } # fm
CONVERT = False
hbar_c = 197.3269804 # MeV.fm

def main(n=6):
    data_paths = next(os.walk("data/"))[1]
    for data_path in data_paths:
        plot("data/" + data_path + "/", n)

def plot(raw_path, n):
    nterms = range(1, n+1)

    tmins = next(os.walk(raw_path))[1]
    for k, v in SPACE_VALS.items():
        if k in raw_path:
            l_space = v
            l_size = k
    
    for tmin in tmins:
        dat_path = raw_path + tmin + "/"
        results = {"K":gv.gload(dat_path + "outputK.json"), "D":gv.gload(dat_path + "outputD.json"), "Fit":gv.gload(dat_path + "outputFit.json")}
        
        fig, axs = plt.subplots(4,3, figsize=(12,14), constrained_layout=True)

        fig.suptitle(f"Plots of ground state parameters and fitting quality\nfor tmin={tmin} and a={l_space} fm", fontsize=24)
        
        x = 0
        y = 0
        for flavour in results:
            datas = results[flavour]
            for param in datas:
                data = datas[param]
                ax = axs[y,x]
                title = flavour + " " + param
                means, errs = split_gvars(data)

                if "Fit" in title:
                    fit_str = ""
                    for n in range(N_EXCL):
                        fit_str = "\n".join((fit_str, "n=" + str(n+1) + " y=" + f"{means[n]:.4}"))
                    ax.text(0.05, 0.05, fit_str, transform=ax.transAxes, fontsize=12)
                    ranges = {
                            "chi2/dof": (0, 1.5),
                            "Q": (0, 1.5),
                            "log(GBF)": (0, 3000)}
                    ax.set_ylim(ranges[param][0], ranges[param][1])
                else:
                    # If this plot is for energy, then convert dE -> E
                    regex = re.compile(r"Eo?_\d+")
                    match = regex.search(title)
                    if match:
                        title = match.group()
                        means = np.exp(means) 
                        if CONVERT:
                            ax.set_ylabel(f"E (MeV)", fontsize=12)
                            means = hbar_c * means / l_space # Converting from lattice units to MeV
                        else:
                            ax.set_ylabel(f"E (lattice units)", fontsize=12)
                        errs = means * errs # Approximate error dE +- err -> E (1 +- err) for err << dE, which is true in all cases studied.
                    good_n = GOOD_VALS[int(tmin)]-1
                    val_str = "y=" + fr"{means[good_n]:.4f} $\pm$ {errs[good_n]:.4f}"
                    ax.text(0.4, 0.8, val_str, transform=ax.transAxes, fontsize=12)

                ax.errorbar(nterms, means, errs, fmt="x", ecolor="r", elinewidth=1, capsize=3)
                ax.set_title(title, fontsize=18)
                ax.set_xlabel("n", fontsize = 12)
                ax.tick_params("both", labelsize=12)
                
                y+=1
            y=0
            x+=1 
                
        # plt.show()
        out_path = "images/" + ("MeV/" if CONVERT else "lattice/") + raw_path.split("/")[1] + "/"
        if not os.path.exists(out_path): os.makedirs(out_path)
        fig.savefig(out_path + l_size + "_tmin" + tmin, pad_inches=0)


def split_gvars(vals):
    means = []
    errs = []
    for val in vals:
        means.append(val.mean)
        errs.append(val.sdev)
    means = np.array(means)
    errs = np.array(errs)
    return (means, errs)

if __name__ == '__main__':
    main()
