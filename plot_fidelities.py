from qutip import *
import numpy as np
import scipy
import itertools
import random
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
k = 2000

q1 = 0
q2 = 2

#W state of N qubits
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

num_of_pauli_setups = int(simulate_pauli.num_of_setups(n))

step = num_of_pauli_setups
start = num_of_pauli_setups

step_pauli = 1
start_pauli = 1


#Number of runs
num_of_runs = 12

args = [(n, k, q1, q2, w_state, start, step, False) for x in range(num_of_runs)]

args_pauli = [(n, k, q1, q2, w_state, start_pauli, step_pauli, False) for x in range(num_of_runs)]

def star_sic(args):
	simulate_sic.main(*args)
	return "Done"

def star_pauli(args):
	simulate_pauli.main(*args)
	return "Done"



if __name__ == "__main__":

	try:
		with open(r'Results\results_sic.pkl', 'rb') as f:
			results = pickle.load(f)
	except:
		p = Pool()
		for i, simulation in enumerate(p.imap_unordered(star_sic, args, 1)):
			print("SIC SIMULATION ROUND {} OF {}".format(i, num_of_runs))
			
		with open(r'Results\results_sic.pkl', 'rb') as f:
			results = pickle.load(f)

	num_of_runs = len(results)

	fids = [x[1] for x in results]
	#print(fids)



	average_fid = np.zeros(len(results[0][0]))
	for i in range(len(fids)):
		for j in range(len(fids[i])):
			average_fid[j] += fids[i][j] / num_of_runs


	k_indexes = results[0][0]


	try:
		with open(r'Results\results_pauli.pkl', 'rb') as f:
			results_pauli = pickle.load(f)
	except:
		p = Pool()
		for i, simulation in enumerate(p.imap_unordered(star_pauli, args_pauli, 1)):
			print("PAULI SIMULATION ROUND {} OF {}".format(i, num_of_runs))
			
		with open(r'Results\results_pauli.pkl', 'rb') as f:
			results_pauli = pickle.load(f)

	num_of_runs_pauli = len(results_pauli)

	fids_pauli = [x[1] for x in results_pauli]
	#print(fids)



	average_fid_pauli = np.zeros(len(results_pauli[0][0]))
	for i in range(len(fids_pauli)):
		for j in range(len(fids_pauli[i])):
			average_fid_pauli[j] += fids_pauli[i][j] / num_of_runs_pauli


	k_indexes_pauli = results_pauli[0][0] * num_of_pauli_setups

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



	plt.xlabel("Number of measurements")
	plt.ylabel("Fidelity")
	plt.title("Average of {} runs for {} qubits, fidelity for qubits {} and {}".format(num_of_runs, n, q1, q2))
	plt.legend(loc='lower right')

	plt.show()
