from qutip import *
import numpy as np
import sys
import scipy
from scipy import stats
import itertools
import random
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import matplotlib
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
import simulate_sic
import simulate_pauli
#use tex
matplotlib.rc('text', usetex = True)

np.seterr(all='ignore')

#Parameters for programs

#Number of qubits
n = 3

#Number of copies
k = 300
#Number of runs
num_of_runs = 5

q1 = 1
q2 = 2

state_name = "GHZ" #GHZ of W

#new plots
new_plots = True
project_path = os.path.abspath(os.getcwd())
simulation_files = [f for f in listdir(project_path + r'\Results') if isfile(join(project_path + r'\Results', f))]

pauli_path = "results_{}_{}_{}_{}_{}_{}_pauli.pkl".format(n, k, num_of_runs, q1, q2, state_name)
sic_path = "results_{}_{}_{}_{}_{}_{}_sic.pkl".format(n, k, num_of_runs, q1, q2, state_name)

if not new_plots:
	for file in simulation_files:
		if 'pauli' in file:
			pauli_path = file
			params = file.split("_")
			n = int(params[1])
			k = int(params[2])
			num_of_runs = int(params[3])
			q1 = int(params[4])
			q2 = int(params[5])
			state_name = params[6]
		else:
			sic_path = file
			params = file.split("_")
			n = int(params[1])
			k = int(params[2])
			num_of_runs = int(params[3])
			q1 = int(params[4])
			q2 = int(params[5])
			state_name = params[6]

num_of_pauli_setups = int(simulate_pauli.num_of_setups(n))

step = num_of_pauli_setups
start = num_of_pauli_setups

step_pauli = 1
start_pauli = 1

#W state of N qubits
def w_state(n):
	w_vec = []
	for i in range(n):
		vec = []
		for j in range(n):
			if i == j:
				vec.append(basis(2, 0))
			else:
				vec.append(basis(2, 1))
		#print(tensor(vec))
		w_vec.append(tensor(vec))		
	w_vec = sum(w_vec).unit()

	#Density matrix for W state
	w_state = w_vec * w_vec.dag()
	return w_state

#GHZ state of N qubits
def GHZ(n):
	ghz_vec = []
	ghz_vec.append(tensor([basis(2, 1) for x in range(n)]))
	ghz_vec.append(tensor([basis(2, 0) for x in range(n)]))
			
	ghz_vec = sum(ghz_vec).unit()

	#Density matrix for GHZ state
	ghz_state = ghz_vec * ghz_vec.dag()
	return ghz_state

input_state = ""
if state_name == "GHZ":
	input_state = GHZ(n)
elif state_name == "W":
	input_state = w_state(n)

args = [(n, k, q1, q2, input_state, start, step, state_name, num_of_runs, False) for x in range(num_of_runs)]

args_pauli = [(n, k, q1, q2, input_state, start_pauli, step_pauli, state_name, num_of_runs, False) for x in range(num_of_runs)]

def star_sic(args):
	simulate_sic.main(*args)
	return "Done"

def star_pauli(args):
	simulate_pauli.main(*args)
	return "Done"



if __name__ == "__main__":

	if not new_plots:
		try:
			with open(r'Results\{}'.format(sic_path), 'rb') as f:
				results = pickle.load(f)
		except:
			p = Pool()
			for i, simulation in enumerate(p.imap_unordered(star_sic, args, 1)):
				print("SIC SIMULATION ROUND {} OF {}".format(i, num_of_runs))
			p.terminate()
			with open(r'Results\{}'.format(sic_path), 'rb') as f:
				results = pickle.load(f)
	else:
		try:
			copyfile(r'Results\{}'.format(sic_path), r'Results\Old\{}'.format(sic_path))
			os.remove(r'Results\{}'.format(sic_path))
		except:
			pass
		p = Pool()
		for i, simulation in enumerate(p.imap_unordered(star_sic, args, 1)):
			print("SIC SIMULATION ROUND {} OF {}".format(i, num_of_runs))
		p.terminate()
		with open(r'Results\{}'.format(sic_path), 'rb') as f:
			results = pickle.load(f)		

	num_of_runs = len(results)

	fids = [x[1] for x in results]
	
	average_fid = np.zeros(len(results[0][0]))
	for i in range(len(fids)):
		for j in range(len(fids[i])):
			average_fid[j] += fids[i][j] / num_of_runs

	#Standard deviation
	std = []
	for j in range(len(fids[0])):
		fids_j = [fids[i][j] for i in range(len(fids))]
		std.append(stats.sem(fids_j))

	k_indexes = results[0][0]

	if not new_plots:
		try:
			with open(r'Results\{}'.format(pauli_path), 'rb') as f:
				results_pauli = pickle.load(f)
		except:
			p = Pool()
			for i, simulation in enumerate(p.imap_unordered(star_pauli, args_pauli, 1)):
				print("PAULI SIMULATION ROUND {} OF {}".format(i, num_of_runs))
			p.terminate()
			with open(r'Results\{}'.format(pauli_path), 'rb') as f:
				results_pauli = pickle.load(f)
	else:
		try:
			copyfile(r'Results\{}'.format(pauli_path), r'Results\Old\{}'.format(pauli_path))
			os.remove(r'Results\{}'.format(pauli_path))
		except:
			pass
		p = Pool()
		for i, simulation in enumerate(p.imap_unordered(star_pauli, args_pauli, 1)):
			print("PAULI SIMULATION ROUND {} OF {}".format(i, num_of_runs))
		p.terminate()
		with open(r'Results\{}'.format(pauli_path), 'rb') as f:
			results_pauli = pickle.load(f)

	num_of_runs_pauli = len(results_pauli)

	fids_pauli = [x[1] for x in results_pauli]


	print(fids)
	print(len(fids))
	print(len(fids[1]))

	print(fids_pauli)
	print(len(fids_pauli))
	print(len(fids_pauli[0]))

	for i in range(len(fids_pauli)):
		print(len(fids_pauli[i]))

	average_fid_pauli = np.zeros(len(results_pauli[0][0]))
	for i in range(len(fids_pauli)):
		for j in range(len(fids_pauli[i])):
			average_fid_pauli[j] += fids_pauli[i][j] / num_of_runs_pauli

	#Standard deviation pauli
	std_pauli = []
	for j in range(len(fids_pauli[0])):
		fids_j = [fids_pauli[i][j] for i in range(len(fids_pauli))]
		std_pauli.append(stats.sem(fids_j))


	k_indexes_pauli = results_pauli[0][0] * num_of_pauli_setups

	print(k_indexes)
	print(k_indexes_pauli)

	#Plots
	#SIC-POVM
	plt.scatter(k_indexes, average_fid, c='b', label="SIC-POVM")

	#Function for fitting
	def func(x, a, b, c):
		return a - b / np.log(c * x)
	
	popt, pcov = scipy.optimize.curve_fit(func, k_indexes, average_fid, 
											bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)))
	print(popt)
	print(pcov)
	#fit = np.poly1d(np.polyfit(k_indexes, np.log(fids), 1, w=np.sqrt(fids)))
	plt.plot(k_indexes, func(k_indexes, *popt))

	plt.errorbar(k_indexes, average_fid, yerr=std, color="blue")
	
	#PAULI MEASUREMENT
	plt.scatter(k_indexes_pauli, average_fid_pauli, c='r', label="Pauli setup")

	real_k_indexes_pauli = k_indexes_pauli
	real_average_fid_pauli = average_fid_pauli
	
	success = False
	while not success:
		try:
			popt, pcov = scipy.optimize.curve_fit(func, k_indexes_pauli, average_fid_pauli, 
													bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)))
			success = True
		except:
			k_indexes_pauli = k_indexes_pauli[1:]
			average_fid_pauli = average_fid_pauli[1:]

	print(popt)
	print(pcov)
	#fit = np.poly1d(np.polyfit(k_indexes, np.log(fids), 1, w=np.sqrt(fids)))
	plt.plot(real_k_indexes_pauli, func(real_k_indexes_pauli, *popt))
	
	plt.errorbar(real_k_indexes_pauli, average_fid_pauli, yerr=std_pauli, color="red")
	
	plt.xlabel("Number of measurements")
	plt.ylabel("Fidelity")
	plt.title("Average of {} runs for {} qubits, fidelity for qubits {} and {} for the {} state".format(num_of_runs, n, q1, q2, state_name))
	plt.legend(loc='lower right')

	plt.show()
	sys.exit()