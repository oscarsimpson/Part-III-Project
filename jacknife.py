import numpy as np
import gvar as gv
import matplotlib.pyplot as plt

time_slices = 96

def jacknife(arr):
	out = np.empty_like(arr)
	num_configs = len(arr)
	print(f'num_configs = {num_configs}')
	num_slices = len(arr[0])
	assert num_slices == time_slices, f'Provided data contains {num_slices} time intervals, {time_slices} expected.'
	print(f'num_slices = {num_slices}') 	
	
	for i in range(num_configs):
		reduced_arr = np.delete(arr, [i], 0)
		out[i] = reduced_arr.mean(0)
	return out	
		

with open("data/2pt_hisq_msml5_fine_D_nongold_489conf.gpl", "r") as textFile:
	raw_data = np.loadtxt(textFile, usecols=range(1,time_slices+1))
	data = jacknife(raw_data)	
	avg = data.mean(0)
	stddev = (100000*data).std(0)

	avg = np.append(avg, avg[0])
	stddev = np.append(stddev, stddev[0])

	fig, ax = plt.subplots(figsize=(12, 7.2))
	ax.errorbar(range(time_slices+1), avg, yerr=stddev, capsize=2, ecolor='k', elinewidth=0.8, fmt='b+', linewidth=0)

	ax.set_title("Average 2-point correlator for n=489 gauge configurations\non a fine lattice", fontsize=24)
	ax.set_xlabel("Time", fontsize=18)
	ax.set_ylabel("Correlation", fontsize=18)
	ax.text(16, -0.1, 'Average over Jacknife sampling was used.\nFirst value duplicated to explicitly show periodicity.\nError bars magnified by $10^5$.', fontsize=16)
	ax.tick_params(axis='both', which='both', labelsize=16)
	ax.set_xticks(np.linspace(0, time_slices, 13))

	plt.savefig("Correlator.png")



