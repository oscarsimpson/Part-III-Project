import numpy as np
import random

time_slices = 96

def make_conf(m):
	out = np.empty(time_slices +1)
	meff = m*1+(random.random()-0.5)/20
	cons = 1+(random.random()-0.5)/20

	out[0] = 0
	for i in range(time_slices):
		out[i+1] = cons*(np.exp(-meff*i) + np.exp(-meff*(time_slices-i)))
	
	return out

out = []
for i in range(512):
	out.append(make_conf(0.8))

with open("data/testData.gpl", 'w') as f:
	for i in range(len(out)):
		for j in range(len(out[0])):
			f.write(str(out[i][j]))
			f.write(" ")
		f.write("\n")


