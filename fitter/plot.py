import matplotlib.pyplot as plt
import gvar as gv 
import numpy as np

def main(n=6):
    nterms = range(1, n+1)
    
    results = {"K":gv.gload("data/outputK.json"), "D":gv.gload("data/outputD.json"), "Fit":gv.gload("data/outputFit.json")}
    
    fig, ax = plt.subplots(4,3, sharex=True, figsize=(14,12))
    
    x = 0
    y = 0
    for flavour in results:
        datas = results[flavour]
        for param in datas:
            data = datas[param]
            title = flavour + " " + param
            print(x, y, title, data)
            means, errs = split_gvars(data)
            ax[y,x].errorbar(nterms, means, errs)
            ax[y,x].set(title=title)
            y+=1
        y=0
        x+=1 
            
    plt.show()


def split_gvars(vals):
    means = []
    errs = []
    for val in vals:
        means.append(val.mean)
        errs.append(val.sdev)
    return (means, errs)

if __name__ == '__main__':
    main()
