import numpy as np
import scipy
from scipy import stats
import os
from os import listdir
from os.path import isfile, join
import pickle
import matplotlib
import matplotlib.pyplot as plt
import simulate_pauli


#Function for fitting
def func(x, a, b, c):
	return (x-a) / (b + (x-a)**c)**(1 / c)

project_path = os.path.abspath(os.getcwd())
simulation_files = [f for f in listdir(project_path + r'\Results') if isfile(join(project_path + r'\Results', f))]

#simulation_files = [x for x in simulation_files if "GHZ" in x]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for col, file in enumerate(simulation_files):
	with open(r'Results\{}'.format(file), 'rb') as f:
		results = pickle.load(f)

	if "pauli." in file:
		n = int(file.split("_")[1])
		num_of_pauli_setups = int(simulate_pauli.num_of_setups(n))


	num_of_runs = len(results)
	fids = [x[1] for x in results]
	average_fid = np.zeros(len(results[0][0]))
	for i in range(len(fids)):
		for j in range(len(fids[i])):
			average_fid[j] += fids[i][j] / num_of_runs

	#Standard deviation
	if len(fids) == 1:
		std = fids
	else:
		std = []
		for j in range(len(fids[0])):
			fids_j = [fids[i][j] for i in range(len(fids))]
			std.append(stats.sem(fids_j))

	if len(fids) == 1:
		sigma = fids
	else:
		sigma = []
		for j in range(len(fids[0])):
			fids_j = [fids[i][j] for i in range(len(fids))]
			sigma.append(np.std(fids_j))

	k_indexes = results[0][0]
	if "pauli." in file:
		k_indexes = results[0][0] * num_of_pauli_setups

	plt.figure(0)
	plt.scatter(k_indexes, average_fid, label=file, color=colors[col % len(colors)])

	real_k_indexes = k_indexes
	real_fids = average_fid

	success = False
	while not success:
		try:
			if len(fids) > 1:
				popt, pcov = scipy.optimize.curve_fit(func, k_indexes, average_fid, sigma = std)
			else:
				popt, pcov = scipy.optimize.curve_fit(func, k_indexes, average_fid)
			success = True
			found_fit = True
		except:
			k_indexes = k_indexes[1:]
			average_fid = average_fid[1:]
			if len(k_indexes) == 0:
				success = True
				found_fit = False
	print(file)
	print(popt)
	print(pcov)
	if found_fit:
		plt.plot(range(15, real_k_indexes[-1]), func(range(15, real_k_indexes[-1]), *popt), label="Fit to {}".format(file), color=colors[col % len(colors)])
	if len(fids) > 1:
		plt.errorbar(real_k_indexes, real_fids, yerr=std, color=colors[col % len(colors)])
	
	plt.figure(1)
	if found_fit:
		plt.plot(real_k_indexes, func(real_k_indexes, *popt) - real_fids, 
				label="Difference to fit: {}".format(file), color=colors[col % len(colors)])


plt.plot(real_k_indexes, [0 for x in range(len(real_k_indexes))], color='k')
plt.figure(0)
plt.plot(real_k_indexes, [1 for x in range(len(real_k_indexes))], color='k')
plt.plot(real_k_indexes, [0.99 for x in range(len(real_k_indexes))], color='dimgray')
plt.plot(real_k_indexes, [0.98 for x in range(len(real_k_indexes))], color='dimgrey')
plt.plot(real_k_indexes, [0.97 for x in range(len(real_k_indexes))], color='gray')
plt.plot(real_k_indexes, [0.96 for x in range(len(real_k_indexes))], color='grey')
plt.plot(real_k_indexes, [0.95 for x in range(len(real_k_indexes))], color='darkgray')
plt.xlabel("Number of copies of state")
plt.ylabel("Fidelity")
plt.legend(loc='lower right')
plt.figure(1)
plt.xlabel("Number of copies of state")
plt.ylabel("Difference between fit and average")

plt.legend(loc='lower right')

plt.show()