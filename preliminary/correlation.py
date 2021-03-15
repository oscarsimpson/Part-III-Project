import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # Parse cmd-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    args = parser.parse_args()
    filename = args.filename if args.filename else "2pt_hisq_msml5_fine_D_nongold_489conf"
    # Load data
    data = np.loadtxt(f"data/{filename}.dat")

    plot(data, f"Average 2-point correlator for n={len(data)} gauge configurations\non a fine lattice", filename=f"correlation_{filename}")

def plot(correlation_data, title, filename):
    avg = correlation_data.mean(0)
    stddev = correlation_data.std(0)
    time_slices = len(avg)

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_title(title, fontsize=24)

    ax.plot(range(time_slices), avg, 'b+', linewidth=0)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Correlation", fontsize=18)
    ax.tick_params(axis='both', which='both', labelsize=16)

    ax2 = ax.twinx()
    ax2.plot(stddev, color='orange', linewidth=0.8)
    ax2.set_yscale('log')
    ax2.set_ylabel('Error', fontsize=18)
    ax2.tick_params(axis='both', which='both', labelsize=16)

    fig.tight_layout()
    plt.savefig(f"images/{filename}.png")

main()
