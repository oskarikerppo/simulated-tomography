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
import simulate_povm
import simulate_pauli
#use tex
matplotlib.rc('text', usetex = True)

np.seterr(all='ignore')


#Pauli matrices
s = [sigmax(), sigmay(), sigmaz()]

#General qubit state, input as list of Bloch vector components, i.e. r = [rx, ry, rz]
def rho(r):
	if np.linalg.norm(r) != 1:
		r = np.array(r)/np.linalg.norm(r)
	return (qeye(2) + sum([r[i] * s[i] for i in range(3)])) / 2

project_path = os.path.abspath(os.getcwd())
simulation_files = [f for f in listdir(project_path + r'\Results') if isfile(join(project_path + r'\Results', f))]

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


def star_povm(args):
	simulate_povm.main(*args)
	return "Done"

def star_pauli(args):
	simulate_pauli.main(*args)
	return "Done"



def main(n, k, q1, q2, input_state, POVM, start, step, state_name, meas_name, num_of_runs, seed=False):
	try:
		for file in simulation_files:
			copyfile(r'Results\{}'.format(file), r'Results\Old\{}'.format(file))
			os.remove(r'Results\{}'.format(file))
	except:
		pass

	if meas_name == "pauli":
		args = [(n, k, q1, q2, input_state, start, step, state_name, num_of_runs, seed) for x in range(num_of_runs)]
		p = Pool()
		for i, simulation in enumerate(p.imap_unordered(star_pauli, args, 1)):
			print("pauli SIMULATION ROUND {} OF {}".format(i, num_of_runs))
		p.terminate()
	else:
		args = [(n, k, q1, q2, input_state, POVM, start, step, state_name, meas_name, num_of_runs, seed) for x in range(num_of_runs)]
		p = Pool()
		for i, simulation in enumerate(p.imap_unordered(star_povm, args, 1)):
			print("{} SIMULATION ROUND {} OF {}".format(meas_name, i, num_of_runs))
		p.terminate()


#SIC-POVM for qubit
E1 = rho([1,1,1])/2
E2 = rho([1,-1,-1])/2
E3 = rho([-1,1,-1])/2
E4 = rho([-1,-1,1])/2

E = [E1, E2, E3, E4]

PX = [s[0].eigenstates()[1][0] * s[0].eigenstates()[1][0].dag(),
	  s[0].eigenstates()[1][1] * s[0].eigenstates()[1][1].dag()]

PY = [s[1].eigenstates()[1][0] * s[1].eigenstates()[1][0].dag(),
	  s[1].eigenstates()[1][1] * s[1].eigenstates()[1][1].dag()]

PZ = [s[2].eigenstates()[1][0] * s[2].eigenstates()[1][0].dag(),
	  s[2].eigenstates()[1][1] * s[2].eigenstates()[1][1].dag()]

pauli_povm = [PX[0]/3, PX[1]/3, PY[0]/3, PY[1]/3, PZ[0]/3, PZ[1]/3]

noise_param = 0.25

noisy_sic = [noise_param*qeye(2)/4 + (1-noise_param)*E[i] for i in range(len(E))]



#Parameters for programs

#Number of qubits
n = 3

#Number of copies
k = 1500
#Number of runs
num_of_runs = 25

q1 = 0
q2 = 0

state_name = "GHZ" #GHZ of W

input_state = ""
if state_name == "GHZ":
	input_state = GHZ(n)
elif state_name == "W":
	input_state = w_state(n)


#Average fidelity
calculate_average = True
if calculate_average:
	q1 = 0
	q2 = 0

num_of_pauli_setups = int(simulate_pauli.num_of_setups(n))

meas_name = "pauli_povm" #sic_povm, noisy_sic, pauli_povm or pauli
POVM = pauli_povm

if meas_name == "pauli":
	start = 1
	step = 1
else:
	step = num_of_pauli_setups
	start = num_of_pauli_setups




if __name__ == "__main__":
	#main(n, k, q1, q2, input_state, POVM, start, step, state_name, meas_name, num_of_runs, seed=False)
	main(3, k, q1, q2, GHZ(3), 		POVM, int(simulate_pauli.num_of_setups(3)), int(simulate_pauli.num_of_setups(3)), "GHZ", 	meas_name, num_of_runs, seed=False)
	main(3, k, q1, q2, w_state(3), 	POVM, int(simulate_pauli.num_of_setups(3)), int(simulate_pauli.num_of_setups(3)), "W", 		meas_name, num_of_runs, seed=False)
	main(4, k, q1, q2, GHZ(4), 		POVM, int(simulate_pauli.num_of_setups(4)), int(simulate_pauli.num_of_setups(4)), "GHZ", 	meas_name, num_of_runs, seed=False)
	main(4, k, q1, q2, w_state(4), 	POVM, int(simulate_pauli.num_of_setups(4)), int(simulate_pauli.num_of_setups(4)), "W", 		meas_name, num_of_runs, seed=False)
	sys.exit()