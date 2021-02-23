import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import argparse

def main():
    # Parse cmd-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    parser.add_argument("-t", "--time_slices", help="Must match the number of time slices in the provided gpl file")
    args = parser.parse_args()
    filename = args.filename if args.filename else "2pt_hisq_msml5_fine_D_nongold_489conf"
    time_slices = int(args.time_slices if args.time_slices else 96)
    # Load data
    data = np.loadtxt(f"data/{filename}.dat")

    with open(f"data/{filename}.gpl", "r") as textFile:
        raw_data = np.loadtxt(textFile, usecols=range(1,time_slices+1))
        data = jacknife(raw_data)	
        np.savetxt(f"data/{filename}.dat", data)

def jacknife(arr):
    out = np.empty_like(arr)
    num_configs = len(arr)
    print(f'num_configs = {num_configs}')
    num_slices = len(arr[0])
    print(f'num_slices = {num_slices}') 	

    for i in range(num_configs):
        reduced_arr = np.delete(arr, [i], 0)
        out[i] = reduced_arr.mean(0)
    return out	

main()
