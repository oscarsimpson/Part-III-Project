import json
import gvar as gv
import lsqfit
import numpy as np
import matplotlib.pyplot as plt
import sys

OUT_TO_FILE = True
if OUT_TO_FILE:
    sys.stdout = open("out.dat", "w")

ORDER = ["K", "D"]
m_q = [93, 1270] #MeV # P.A. Zyla et al. (Particle Data Group), Prog. Theor. Exp. Phys. 2020, 083C01 (2020)
m_meson = [493.611, 1864.83] #MeV # Same source
alpha = 0.007297 # approx 1/137 # Same source
Gamma_const = 4*np.pi/3 * alpha**2 * (1/3)**2

LAMBDA = 500 #MeV # Sets the energy scale # Chakraborty et al. arXiv e-print 1703.05552 (2017)

def main():
    with open("input.json", "r") as f:
        datas = json.loads(f.read())
    
    for k, v in datas.items():
        datas[k] = make_gvar(v, ["E0", "a0"])
    
    E0s = {}
    # E0s is a dictionary of len2 arrays of gvars
    for k, v in datas.items():
        E0s[k] = v["E0"]
    
    Gs = {}
    fs = {}

    for k, v in datas.items():
        fs[k] = make_fs(v)
        v["f"] = fs[k]

    E0fit = compute_fit(datas, E0s, "E")
    ffit = compute_fit(datas, fs, "a")
    
    G = Gamma_const * model({"Extrapolated":{"a":0, "mlms":0}}, ffit.p)["Extrapolated"]**2 / m_meson
    print("=========")
    print(f"Gamma_K = {G[0]} MeV")
    print(f"Gamma_D = {G[1]} MeV")

    plot_E0(datas, E0fit.p)
    plot_f(datas, ffit.p, False)

def plot_E0(data, params):
    data=data.copy()
    data["literature"] = {"a":0, "mlms":0, "E0":[gv.gvar(497.611, 0.013), gv.gvar(1864.83, 0.05)]}
    scale = 0.16**2
    a2s = np.arange(100)
    fig, axs = plt.subplots(1, 2, figsize=(9,4), constrained_layout=True)
    
    inputs = {}
    for x in a2s:
        inputs[x] = {"a":np.sqrt(x*scale/100), "mlms":0}

    ca2s = params["ca2"]
    minca2s = np.asarray([val.mean-val.sdev for val in ca2s])
    maxca2s = np.asarray([val.mean+val.sdev for val in ca2s])
    As = params["A"]
    minAs = np.asarray([val.mean-val.sdev for val in As])
    maxAs = np.asarray([val.mean+val.sdev for val in As])

    minparams = dict(params)
    minparams["ca2"] = minca2s
    minparams["A"] = minAs
    maxparams = dict(params)
    maxparams["ca2"] = maxca2s
    maxparams["A"] = maxAs
    
    minfit = sanitise_data(model(inputs, minparams))
    midfit = sanitise_data(model(inputs, params))
    maxfit = sanitise_data(model(inputs, maxparams))

    for i in range(2):
        ax=axs[i]
        ax.plot(a2s*scale/100, minfit[i], "m--", ms=0, label=r"$\pm\sigma$ interval")
        ax.plot(a2s*scale/100, midfit[i], "k:" , ms=0)
        ax.plot(a2s*scale/100, maxfit[i], "m--", ms=0)
        
        for k in data:
            v = data[k]
            ax.errorbar(v["a"]**2, v["E0"][i].mean, v["E0"][i].sdev, fmt="x", label=f"{k} lattice", capsize=3)

        if i==0:
            ax.legend()
        ax.set_title(fr"${ORDER[i]}$ meson")
        ax.set_xlabel(r"$a^2 (fm^2)$", fontsize=12)
        ax.set_ylabel(r"$E_0$ (MeV)", fontsize=12)

    fig.savefig(f"images/ccextrapolate_E0.png")

def make_fs(v):
    m = m_q+v['mlms']*m_q[0]
    return m * gvar_abs(v['a0']) * np.sqrt(2 / v['E0']**3)

def plot_f(data, params, plot_fit=True):

    data=data.copy()
    scale = 0.16**2
    a2s = np.arange(100)
    fig, axs = plt.subplots(1, 2, figsize=(9,4), constrained_layout=True)
    
    inputs = {}
    for x in a2s:
        inputs[x] = {"a":np.sqrt(x*scale/100), "mlms":0}

    ca2s = params["ca2"]
    minca2s = np.asarray([val.mean-val.sdev for val in ca2s])
    maxca2s = np.asarray([val.mean+val.sdev for val in ca2s])
    As = params["A"]
    minAs = np.asarray([val.mean-val.sdev for val in As])
    maxAs = np.asarray([val.mean+val.sdev for val in As])

    minparams = dict(params)
    minparams["ca2"] = minca2s
    minparams["A"] = minAs
    maxparams = dict(params)
    maxparams["ca2"] = maxca2s
    maxparams["A"] = maxAs
    
    minfit = sanitise_data(model(inputs, minparams))
    midfit = sanitise_data(model(inputs, params))
    maxfit = sanitise_data(model(inputs, maxparams))

    for i in range(2):
        ax=axs[i]
        if plot_fit:
            ax.plot(a2s*scale/100, minfit[i], "m--", ms=0, label=r"$\pm\sigma$ interval")
            ax.plot(a2s*scale/100, midfit[i], "k:" , ms=0)
            ax.plot(a2s*scale/100, maxfit[i], "m--", ms=0)
        
        for k in data:
            v = data[k]
            ax.errorbar(v["a"]**2, v["f"][i].mean, v["f"][i].sdev, fmt="x", label=f"{k} lattice", capsize=3)

        if i==0:
            ax.legend()
        ax.set_title(fr"${ORDER[i]}$ meson")
        ax.set_xlabel(r"$a^2 (fm^2)$", fontsize=12)
        ax.set_ylabel(rf"$f_{ORDER[i]}$", fontsize=12)

    fig.savefig(f"images/ccextrapolate_f.png")

def sanitise_data(d):
    # d is a dict of N arrays of [gvar, gvar]
    # return [[float, ... N],[float, ... N]] 
    out = [0 for _ in d]
    for k, v in d.items():
        out[k] = np.asarray([val.mean for val in v])
    return np.asarray(out).T

def gvar_abs(vals):
    return np.asarray([gv.gvar(np.abs(val.mean), val.sdev) for val in vals])
    
def compute_fit(datas, ys, var):
    fit = lsqfit.nonlinear_fit(data = (datas, ys), prior=build_prior(var), fcn=model)
    
    print("=========")
    print_dict(fit.p)
    print(f"chi2={fit.chi2} dof={fit.dof} Q={fit.Q}")
    compare(fit.p, datas, ys)
    extrapolated_ys = model({"Extrapolated": {"a":0, "mlms":0}}, fit.p)
    print()
    print_dict(extrapolated_ys)
    print()
    return fit


def build_prior(var):
    prior={}
    prior['A'] = [gv.gvar(500, 100), gv.gvar(2000, 100)] if var=="E" else [gv.gvar(200, 20), gv.gvar(200, 20)]
    prior['ca2'] = [gv.gvar(0, 0.5), gv.gvar(0, 0.5)] if var=="E" else [gv.gvar(0, 0.5), gv.gvar(0, 0.5)]
    prior['cdelta'] = [gv.gvar(0, 5), gv.gvar(0, 5)] if var=="E" else [gv.gvar(0, 5), gv.gvar(0, 5)]
    return prior

def compare(params, datas, E0s):
    datas_pred = model(datas, params)
    print()
    print("Predicted values")
    print_dict(datas_pred)
    print()
    print("Original values")
    print_dict(E0s)

def print_dict(d):
    for k, v in d.items():
        print(f"{k} = {v}")

def model(datas, p):
    # Given parameters p, compute "y" values (masses) at the points in "datas"
    A = p['A']
    cdelta = p['cdelta']
    ca2 = p['ca2']
    
    out={}
    for k, data in datas.items():
        # k is the name of a dataset, data is the params ("x") for that dataset
        x = data['a']
        z = data['mlms']
        out[k]=A*(1 + cdelta/10 *2*z + ca2*(x*LAMBDA)**2) # out[k] should be an array of size 2

    return out

def make_gvar(data, keys):
    # data is a dict{ **params, **keys}, where each key gives a dict of strings that are to be converted to an array of gvar
    out = {}
    for k, v in data.items():
        if k in keys:
            out[k] = []
            for gk in ORDER:
                # out[k] is a list of gvars in the order given by ORDER
                out[k].append(gv.gvar(v[gk]))
            out[k] = np.asarray(out[k])
        else:
            out[k] = np.asarray(v)
    return out

if __name__ == "__main__":
    main()
