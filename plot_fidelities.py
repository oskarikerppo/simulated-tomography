from qutip import *
import numpy as np
import scipy
import itertools
import random
import matplotlib
import matplotlib.pyplot as plt
import pickle
import simulate_sic



#Parameters for programs
#Number of qubits
n = 3

#Number of copies
k = 675

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


step = 9
start = 9




simulate_sic.main(n, k, q1, q2, w_state, start, step)

die



matplotlib.rc('text', usetex = True)


with open(r'Results\results_sic.pkl', 'rb') as f:
	results = pickle.load(f)

num_of_runs = len(results)

print(num_of_runs)

fids = [x[1] for x in results]
#print(fids)



average_fid = np.zeros(len(results[0][0]))
for i in range(len(fids)):
	for j in range(len(fids[i])):
		average_fid[j] += fids[i][j] / num_of_runs


k_indexes = results[0][0]


with open(r'Results\results_pauli.pkl', 'rb') as f:
	results_pauli = pickle.load(f)

num_of_runs_pauli = len(results_pauli)

print(num_of_runs_pauli)

fids_pauli = [x[1] for x in results_pauli]
#print(fids)



average_fid_pauli = np.zeros(len(results_pauli[0][0]))
for i in range(len(fids_pauli)):
	for j in range(len(fids_pauli[i])):
		average_fid_pauli[j] += fids_pauli[i][j] / num_of_runs_pauli


k_indexes_pauli = results_pauli[0][0] * 9

for i in range(len(results_pauli)):
	if results_pauli[i][0][0] != 2 and results_pauli[i][0][0] != 30:
		print(results_pauli[i][0][0])
		raise 


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

#PAULI MEASUREMENT
plt.scatter(k_indexes_pauli, average_fid_pauli, c='r', label="Pauli setup")

#Function for fitting
def func(x, a, b, c):
	return a - b / np.log(c * x)
 
popt, pcov = scipy.optimize.curve_fit(func, k_indexes_pauli, average_fid_pauli, 
										bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)))
print(popt)
print(pcov)
#fit = np.poly1d(np.polyfit(k_indexes, np.log(fids), 1, w=np.sqrt(fids)))
plt.plot(k_indexes_pauli, func(k_indexes_pauli, *popt))



plt.xlabel("Number of measurements")
plt.ylabel("Fidelity")
plt.title("Average of 3 runs for 3 qubits, fidelity for qubits 0 and 2")
plt.legend(loc='lower right')

plt.show()
