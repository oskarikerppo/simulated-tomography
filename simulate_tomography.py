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
import time
import warnings

#use tex
matplotlib.rc('text', usetex = True)

np.seterr(all='ignore')
warnings.filterwarnings('ignore')

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
		fids = []
		round_time = time.time()
		print("Starting pauli setup {} state...".format(state_name))
		p = Pool(1)
		for i, simulation in enumerate(p.imap_unordered(star_pauli, args, 1)):
			fids.append((i, simulation))
			round_finish = time.time()
			print("pauli setup {} state SIMULATION ROUND {} OF {} in {} minutes".format(state_name, i, num_of_runs, (round_finish - round_time) / 60))
		p.terminate()
		print("Finished {} in {} minutes!".format(meas_name, (round_finish - round_time) / 60))
		return fids
	else:
		args = [(n, k, q1, q2, input_state, POVM, start, step, state_name, meas_name, num_of_runs, seed) for x in range(num_of_runs)]
		fids = []
		round_time = time.time()
		print("Starting {} {} state...".format(meas_name, state_name))
		p = Pool(1)
		for i, simulation in enumerate(p.imap_unordered(star_povm, args, 1)):
			fids.append((i, simulation))
			round_finish = time.time()
			print("{} {} state SIMULATION ROUND {} OF {} in {} minutes".format(meas_name, state_name, i, num_of_runs, (round_finish - round_time) / 60))
		p.terminate()
		print("Finished {} in {} minutes!".format(meas_name, (round_finish - round_time) / 60))
		return fids


#SIC-POVM for qubit
#E1 = rho([1,1,1])/2
#E2 = rho([1,-1,-1])/2
#E3 = rho([-1,1,-1])/2
#E4 = rho([-1,-1,1])/2

#[0.5 / 0 0 0]
#https://en.wikipedia.org/wiki/SIC-POVM
E1 = basis(2, 0) * basis(2, 0).dag() / 2
e2 = basis(2, 0) / np.sqrt(3) + np.sqrt(2/3) * basis(2, 1)
E2 = e2 * e2.dag() / 2
e3 = basis(2, 0) / np.sqrt(3) + np.sqrt(2/3) * np.exp(1j*2*np.pi/3) * basis(2, 1)
E3 = e3 * e3.dag() / 2
e4 = basis(2, 0) / np.sqrt(3) + np.sqrt(2/3) * np.exp(1j*4*np.pi/3) * basis(2, 1)
E4 = e4 * e4.dag() / 2

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
n = 8

#Number of copies
k = 8192

#qiskit copies
#k = 8192

#Number of runs
num_of_runs = 5

q1 = 0
q2 = 0

state_name = "W" #GHZ of W

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

meas_name = "noisy_sic" #sic_povm, noisy_sic, pauli_povm or pauli
POVM = noisy_sic

if meas_name == "pauli":
	start = 1
	step = 1
else:
	step = num_of_pauli_setups
	start = num_of_pauli_setups




if __name__ == "__main__":
	start_time = time.time()

	fid = main(n, k, q1, q2, GHZ(n), 		noisy_sic, 	int(simulate_pauli.num_of_setups(n)), n*int(simulate_pauli.num_of_setups(n)), "W","sic_povm", num_of_runs, seed=False)
	print(fid)
	'''
	#main(n, k, q1, q2, input_state, POVM, start, step, state_name, meas_name, num_of_runs, seed=False)
	start_time_noisy_sic = time.time()
	main(n, k, q1, q2, GHZ(n), 		noisy_sic, 	int(simulate_pauli.num_of_setups(n)), n*int(simulate_pauli.num_of_setups(n)), "GHZ","noisy_sic", num_of_runs, seed=False)
	main(n, k, q1, q2, w_state(n), 	noisy_sic, 	int(simulate_pauli.num_of_setups(n)), n*int(simulate_pauli.num_of_setups(n)), "W", 	"noisy_sic", num_of_runs, seed=False)
	noisy_time = time.time() - start_time_noisy_sic
	print("Finished noisy sic in {} hours, {} minutes and {} seconds!".format(int(np.floor(noisy_time / 3600)), int(np.floor( (noisy_time / 60) % 60)),  int(np.floor(noisy_time % 60))))

	start_time_pauli_povm = time.time()
	main(n, k, q1, q2, GHZ(n), 		pauli_povm, int(simulate_pauli.num_of_setups(n)), n*int(simulate_pauli.num_of_setups(n)), "GHZ","pauli_povm",num_of_runs, seed=False)
	main(n, k, q1, q2, w_state(n), 	pauli_povm, int(simulate_pauli.num_of_setups(n)), n*int(simulate_pauli.num_of_setups(n)), "W", 	"pauli_povm",num_of_runs, seed=False)
	pauli_povm_time = time.time() - start_time_pauli_povm
	print("Finished pauli povm in {} hours, {} minutes and {} seconds!".format(int(np.floor(pauli_povm_time / 3600)), int(np.floor( (pauli_povm_time / 60) % 60)),  int(np.floor(pauli_povm_time % 60))))
	
	start_time_sic_povm = time.time()
	main(n, k, q1, q2, GHZ(n), 		E, int(simulate_pauli.num_of_setups(n)), n*int(simulate_pauli.num_of_setups(n)), 		"GHZ",	"sic_povm"	, num_of_runs, seed=False)
	main(n, k, q1, q2, w_state(n), 	E, int(simulate_pauli.num_of_setups(n)), n*int(simulate_pauli.num_of_setups(n)), 		"W", 	"sic_povm"	, num_of_runs, seed=False)
	sic_povm_time = time.time() - start_time_sic_povm
	print("Finished sic povm in {} hours, {} minutes and {} seconds!".format(int(np.floor(sic_povm_time / 3600)), int(np.floor( (sic_povm_time / 60) % 60)),  int(np.floor(sic_povm_time % 60))))
	
	start_time_pauli_setup = time.time()
	main(n, k, q1, q2, GHZ(n), 		E, 1, 									n,										 		"GHZ",	"pauli"		,num_of_runs, seed=False)
	main(n, k, q1, q2, w_state(n), 	E, 1, 									n, 												"W", 	"pauli"		,num_of_runs, seed=False)
	pauli_time = time.time() - start_time_pauli_setup
	print("Finished pauli setup in {} hours, {} minutes and {} seconds!".format(int(np.floor(pauli_time / 3600)), int(np.floor( (pauli_time / 60) % 60)),  int(np.floor(pauli_time % 60))))

	end_time = time.time()
	total_time = end_time - start_time
	print(total_time)
	print("Finished in {} hours, {} minutes and {} seconds!".format(int(np.floor(total_time / 3600)), int(np.floor( (total_time / 60) % 60)),  int(np.floor(total_time % 60))))
	'''
	sys.exit()