#!/usr/bin/env python
import os
import lsqfit
from corrfitter import Corr2,Corr3,CorrFitter
import numpy as np
from gvar import log,exp,evalcov
from gvar.dataset import Dataset,avg_data
from numpy import array
import gvar as gv

import sys

lsqfit.LSQFit.fmt_parameter = '%8.6f +- %8.6f'

DISPLAYPLOTS = False         # display plots at end of fitting
import matplotlib.pyplot as plt
VERBOSE = 1
DATATAGS = {
        "2pt_hisq_coarse_D_Gold_K_p0_1053conf": {
            "D": "2pt_D_Gold_coarse.ll",
            "K": "2pt_K_coarse_p0.ll"
            },
        "2pt_hisq_msml5_fine_K_zeromom_D_Gold_nongold_495conf": {
            "D": "2pt_D_gold_msml5_fine.ll",
            "K": "2pt_msml5_fine_K_zeromom.ll"
            },
        "2pt_hisq_very-coarse_D_Gold_K_1000conf": {
            "D": "2pt_D_Gold_vc.ll",
            "K": "2pt_K_vc.ll"
            }
        }

TPS = {
        "2pt_hisq_coarse_D_Gold_K_p0_1053conf": 64,
        "2pt_hisq_msml5_fine_K_zeromom_D_Gold_nongold_495conf": 96,
        "2pt_hisq_very-coarse_D_Gold_K_1000conf": 48
        }

ainv = gv.gvar( 1.9006,0.0020)/gv.gvar(0.1715,0.0009)* 0.197326968
#ainv = 1.0/0.12* 0.197326968 
svdcut = 1.0e-3
MAXIT = 10000 # Default is 10,000
# here p0 denotes zero momentum K propagator and Q^2 max,P0 = , p1 = , p2 = , p3 = .... (AS A TWIST)

def main():
    nexp_max = 8
    tmins = [3, 4, 5, 6, 7, 9]
    nterms = range(1, 6+1)

    file_names = next(os.walk("data/"))[2]
    for file_name in file_names:
        data_path = "data/" + file_name.split(".")[0] + "/" # The / is necessary as the compute_data_file function expects this
        compute_data_file(data_path, tmins, nterms)

def compute_data_file(data_path, tmins, nterms):
    data_file = data_path[:-1] + ".gpl"
    dsetfor = Dataset(data_file, binsize=1)
    dset = Dataset()

    ## GPL: not sure what this i
    for key in dsetfor:
      dset[key] = (array(dsetfor[key]))

    data = avg_data(dset)
    print ('Data file: ', data_file)
    
    for tmin in tmins:
        compute_fits(data, nterms, tmin, data_path)
        print("Finished computing for tmin = ", tmin)

def compute_fits(data, nterms, tmin, data_path):
    p0 = data_path + "results"
    if not os.path.exists(data_path): os.makedirs(data_path) # So results is in a extant folder

    model_name = data_path.split("/")[1]
    fitter = CorrFitter(models=build_models(tmin, model_name), ratio=False)

    results = {"K":{
        "a_0":[], "ao_0":[], "log(dE_0)":[], "log(dEo_0)":[]
            }, "D":{
            "a_0":[], "ao_0":[], "log(dE_0)":[], "log(dEo_0)":[]
            }, "Fit":{
            "chi2/dof":[], "Q":[], "log(GBF)":[], 
                }}

    for nexp in nterms:
        fit = fitter.lsqfit(data=data,prior=build_prior(nexp),p0=p0,nterm=nexp,maxit=MAXIT,svdcut=svdcut)
        p = fit.p
        if VERBOSE > 1:
            print()
            print("========"*6)
            print ('nexp =',nexp,' tmin = ',tmin,'svdcut = ',svdcut)
            print("========"*6)
            if VERBOSE > 2:
                print ("----"*3, " PRIOR ", "----"*3)
                print_bufferdict(prior)
                print ("----"*3, "RESULTS", "----"*3)
                print_bufferdict(p)
                print()
            print(fit.format())
            print_results(fit)

        results["K"]["a_0"].append(p["K_max:a"][0])
        results["K"]["ao_0"].append(p["K_max:ao"][0])
        results["K"]["log(dE_0)"].append(p["log(dE.K_max)"][0])
        results["K"]["log(dEo_0)"].append(p["log(dEo.K_max)"][0])

        results["D"]["a_0"].append(p["D_Gold:a"][0])
        results["D"]["ao_0"].append(p["D_Gold:ao"][0])
        results["D"]["log(dE_0)"].append(p["log(dE.D_Gold)"][0])
        results["D"]["log(dEo_0)"].append(p["log(dEo.D_Gold)"][0])

        results["Fit"]["chi2/dof"].append(gv.gvar(fit.chi2/fit.dof,0))
        results["Fit"]["Q"].append(gv.gvar(fit.Q, 0))
        results["Fit"]["log(GBF)"].append(gv.gvar(fit.logGBF, 0))

    out_path = data_path + str(tmin) + "/"
    if not os.path.exists(out_path): os.makedirs(out_path)
    gv.gdump(results["D"], out_path + "outputD.json")
    gv.gdump(results["K"], out_path + "outputK.json")
    gv.gdump(results["Fit"], out_path + "outputFit.json")

    if DISPLAYPLOTS:
        fitter.display_plots()

def build_prior(nexp):
    defaults={"D_Gold:a": [gv.gvar(0.01, 1.0)]*nexp,
            "D_Gold:ao": [gv.gvar(0.01, 0.5)]*nexp,
            "log(dE.D_Gold)": [log(gv.gvar(0.8, 0.3))] + [log(gv.gvar(0.4, 0.2))]*(nexp-1),
            "log(dEo.D_Gold)": [log(gv.gvar(1.1, 0.4))] + [log(gv.gvar(0.4, 0.2))]*(nexp-1),

            "K_max:a": [gv.gvar(0.01, 10.0)]*nexp,
            "K_max:ao": [gv.gvar(0.01, 5.0)]*nexp,
            "log(dE.K_max)": [log(gv.gvar(1.0, 1.0))] + [log(gv.gvar(2.0, 1.0))]*(nexp-1),
            "log(dEo.K_max)": [log(gv.gvar(1.0, 1.0))] + [log(gv.gvar(1.0, 1.0))]*(nexp-1)
            }
    prior = gv.BufferDict(defaults)
    return prior

def print_results(fit):
    dE_D_Gold = exp(fit.p['log(dE.D_Gold)'])
    print ('dE.D_Gold =',fmtlist(dE_D_Gold[:3]))
    print ('E.D_Gold =',fmtlist([sum(dE_D_Gold[:i+1]) for i in range(3)]))
    print ()

    dE_K_max = exp(fit.p['log(dE.K_max)'])
    print ('dE.K_max =',fmtlist(dE_K_max[:3]))
    print ('E.K_max =',fmtlist([sum(dE_K_max[:i+1]) for i in range(3)]))
    print ()

def print_bufferdict(prior):
    for (key, val) in prior.items():
        print(key, " = ", val)

def fmtlist(x):
    return '  '.join([xi.fmt(6) for xi in x]) # 6 decimal places of precision

def build_models(tminD, model_name):
    tminK = 3 # Force tmin=3 for K
    tp = TPS[model_name]
    tdata = range(0,tp)
    tfitD = range(tminD,tp+1-tminD) # all ts
    tfitK = range(tminK,tp+1-tminK) # all ts
    # range(3, Lt+1 -3) for Lt = 96 for K 
    models = [
        Corr2(datatag=DATATAGS[model_name]["K"], tp=tp,tdata=tdata,tfit=tfitK,
            a=('K_max:a','K_max:ao'),b=('K_max:a','K_max:ao'),
            dE=('dE.K_max','dEo.K_max'),s=(1.,-1.)),

        Corr2(datatag=DATATAGS[model_name]["D"], tp=tp,tdata=tdata,tfit=tfitD,
            a=('D_Gold:a','D_Gold:ao'),b=('D_Gold:a','D_Gold:ao'),
            dE=('dE.D_Gold','dEo.D_Gold'),s=(1.,-1.)),

        ]
    return models


def make_data(filename):
    dset = gv.dataset.Dataset(filename)
    return gv.dataset.avg_data(dset)


if __name__ == '__main__':
    main()
