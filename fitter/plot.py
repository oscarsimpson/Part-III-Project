import matplotlib.pyplot as plt
import gvar as gv 
import numpy as np

import os

N_EXCL = 4 # Number of terms that we expect to be fit poorly, and will print their fitting values.
GOOD_VALS = {
        3:4,
        4:4,
        5:4,
        6:4,
        7:4,
        9:4} # for tmin = key, the n = value that we decide is a good enough fit to report a converged parameter

def main(n=6):
    nterms = range(1, n+1)

    tmins = next(os.walk("data/"))[1]
    
    for tmin in tmins:
        dat_path = "data/" + tmin + "/"
        results = {"K":gv.gload(dat_path + "outputK.json"), "D":gv.gload(dat_path + "outputD.json"), "Fit":gv.gload(dat_path + "outputFit.json")}
        
        fig, ax = plt.subplots(4,3, sharex=True, figsize=(14,12))

        fig.suptitle("Plots of ground state parameters and fitting qualitiy for tmin = " + tmin, fontsize=24)
        
        x = 0
        y = 0
        for flavour in results:
            datas = results[flavour]
            for param in datas:
                data = datas[param]
                title = flavour + " " + param
                means, errs = split_gvars(data)
                ns = nterms
                if "Fit" in title:
                    fit_str = ""
                    for n in range(N_EXCL):
                        fit_str = "\n".join((fit_str, "n=" + str(n+1) + " y=" + f"{means[n]:.4}"))
                    ax[y,x].text(0.05, 0.05, fit_str, transform=ax[y,x].transAxes, fontsize=12)
                    ranges = {
                            "chi2/dof": (0, 2),
                            "Q": (0,2),
                            "log(GBF)": (0, 4000)}
                    ax[y, x].set_ylim(ranges[param][0], ranges[param][1])
                else:
                    good_n = GOOD_VALS[int(tmin)]-1
                    val_str = "y=" + fr"{means[good_n]:.4f} $\pm$ {errs[good_n]:.4f}"
                    ax[y, x].text(0.44, 0.85, val_str, transform=ax[y,x].transAxes, fontsize=12)
                ax[y,x].errorbar(ns, means, errs)
                ax[y,x].set(title=title)
                ax[y,x].title.set_size(18)
                
                y+=1
            y=0
            x+=1 
                
        # plt.show()
        plt.savefig("images/tmin" + tmin, pad_inches=0)


def split_gvars(vals):
    means = []
    errs = []
    for val in vals:
        means.append(val.mean)
        errs.append(val.sdev)
    return (means, errs)

if __name__ == '__main__':
    main()
