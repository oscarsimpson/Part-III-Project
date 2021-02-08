import numpy as np
import matplotlib.pyplot as plt

def plot(data, title, ylabel, avg_data=None, filename=None):
	if filename == None:
		filename = ylabel
	avg = data.mean(0)
	stddev = data.std(0)
	time_slices = len(avg)


	fig, ax = plt.subplots(figsize=(12, 7.2))
	ax.set_title(title, fontsize=24)

	ax.plot(range(time_slices), avg, 'b+', linewidth=0)
	if avg_data != None:
		ax.axhline(avg_data, color='blue', linestyle=':') 
	ax.set_xlabel("Time", fontsize=18)
	ax.set_ylabel(ylabel, fontsize=18)
	ax.tick_params(axis='both', which='both', labelsize=16)

	ax2 = ax.twinx()
	ax2.plot(stddev, color='orange', linewidth=0.8)
	ax2.set_yscale('log')
	ax2.set_ylabel('Error', fontsize=18)
	ax2.tick_params(axis='both', which='both', labelsize=16)

	fig.tight_layout()
	plt.savefig(f"images/{filename}.png")
