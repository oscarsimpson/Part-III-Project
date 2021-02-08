import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import sys

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
		

if len(sys.argv) == 1:
	filename = "2pt_hisq_msml5_fine_D_nongold_489conf"
else:
	filename = sys.argv[1]

with open(f"data/{filename}.gpl", "r") as textFile:
	raw_data = np.loadtxt(textFile, usecols=range(1,time_slices+1))
	data = jacknife(raw_data)	
	np.savetxt(f"data/{filename}.dat", data)

