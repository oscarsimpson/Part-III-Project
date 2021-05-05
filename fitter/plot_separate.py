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
CONVERT = True
hbar_c = 197.3269804 # MeV.fm

def main(n=6):
    data_paths = next(os.walk("data/"))[1]
    for data_path in data_paths:
        plot("data/" + data_path + "/", n)

def plot(raw_path, n): # raw_path contains the folder eg data/2pt_hisq_coarse_D_Gold_K_p0_1053conf/
    out_path = "images_separate/" + ("MeV/" if CONVERT else "lattice/") + raw_path.split("/")[1] + "/"
    if not os.path.exists(out_path): os.makedirs(out_path)

    nterms = range(1, n+1)
    tmins = sorted(next(os.walk(raw_path))[1])
    for k, v in SPACE_VALS.items():
        if k in raw_path:
            l_space = v
            l_size = k
    
    # Create plot for the fitting qualityy (only chi2/dof)
    # We hope that there are exactly 6 different tmins (3*2) otherwise this shape will need to be changed
    fits_xmax = 2
    fits_ymax = 3
    figF, axFs = plt.subplots(fits_ymax, fits_xmax, figsize = (7, 9), constrained_layout=True)
    
    for tmin_index in range(len(tmins)):
        tmin = tmins[tmin_index]
        dat_path = raw_path + tmin + "/"
        results = {"K":gv.gload(dat_path + "outputK.json"), "D":gv.gload(dat_path + "outputD.json")}
        fits = gv.gload(dat_path + "outputFit.json")
        
        figD, axDs = plt.subplots(2,2, figsize = (9, 6), constrained_layout=True)
        figK, axKs = plt.subplots(2,2, figsize = (9, 6), constrained_layout=True)
        axs = {"D": axDs, "K": axKs}
        figs = {"D": figD, "K": figK}

        for flavour in results:
            datas = results[flavour]
            for param in datas:
                x = 1 if "o_" in param else 0 # _ is required so log doesn't match
                y = 1 if "a" in param else 0
                data = datas[param]
                ax = axs[flavour][y,x]
                title = param
                means, errs = split_gvars(data)

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
                        ax.set_ylabel(r"E (lattice units)", fontsize=12)
                    errs = means * errs # Approximate error dE +- err -> E (1 +- err) for err << dE, which is true in all cases studied.
                title = r"$" + title + r"$"
                good_n = GOOD_VALS[int(tmin)]-1
                val_str = "y=" + fr"{means[good_n]:.4f} $\pm$ {errs[good_n]:.4f}"
                ax.text(0.4, 0.5, val_str, transform=ax.transAxes, fontsize=12)
                ax.errorbar(nterms, means, errs, fmt="x", ecolor="r", elinewidth=1, capsize=3)
                ax.set_title(title, fontsize=18)
                ax.set_xlabel("n", fontsize=12)
                ax.tick_params("both", labelsize=12)
            # Save figures for K and D plots
            out_name = l_size + "_tmin" + tmin + "_" + flavour
            if flavour=="K" and not tmin=="3":
                pass # For K meson, only tmin==3 is used
            else:
                figs[flavour].savefig(out_path + out_name, pad_inches=0)
            plt.close(figs[flavour]) # To avoid memory issues with making nflavours * 6 * (2*2) figures

        # Plot fitting quality (only chi2/dof)
        ax = axFs[tmin_index // fits_xmax, tmin_index % fits_xmax]
        data = fits["chi2/dof"]
        title = r"$t_\mathrm{min} = " + tmin + r"$"
        means, _ = split_gvars(data)
        fit_str = ""
        for n in range(N_EXCL):
            fit_str = "\n".join((fit_str, "n=" + str(n+1) + " y=" + f"{means[n]:.4}"))
        ax.text(0.03, 0.05, fit_str, transform=ax.transAxes, fontsize=12)
        ax.set_ylim(0.5, 1.5)
        ax.plot(nterms, means, "x")
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("n", fontsize=12)
        ax.tick_params("both", labelsize=12)
    out_name = l_size + "_fitQuality"
    figF.savefig(out_path + out_name, pad_inches=0)

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
