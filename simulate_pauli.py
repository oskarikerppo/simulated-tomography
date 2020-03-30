from qutip import *
import numpy as np
import scipy
import itertools
import random
import matplotlib.pyplot as plt
from itertools import permutations
import pickle

#Pauli matrices
s = [sigmax(), sigmay(), sigmaz()]
s_dict = {'X': sigmax(), 'Y': sigmay(), 'Z': sigmaz()}

PX = [s[0].eigenstates()[1][0] * s[0].eigenstates()[1][0].dag(),
	  s[0].eigenstates()[1][1] * s[0].eigenstates()[1][1].dag()]

PY = [s[1].eigenstates()[1][0] * s[1].eigenstates()[1][0].dag(),
	  s[1].eigenstates()[1][1] * s[1].eigenstates()[1][1].dag()]

PZ = [s[2].eigenstates()[1][0] * s[2].eigenstates()[1][0].dag(),
	  s[2].eigenstates()[1][1] * s[2].eigenstates()[1][1].dag()]

pauli_projection_dict = {
	'X': PX, 'Y': PY, 'Z': PZ
}

#General qubit state, input as list of Bloch vector components, i.e. r = [rx, ry, rz]
def rho(r):
	if np.linalg.norm(r) != 1:
		r = np.array(r)/np.linalg.norm(r)
	return (qeye(2) + sum([r[i] * s[i] for i in range(3)])) / 2

def colorings(n):#n is number of qubits as int
	l = [x for x in range(int(np.ceil(np.log(n)/np.log(3))))]
	cols = []
	for c in l:
		cols.append([int(np.floor(i/(3**c)) % 3) for i in range(n)])
	return cols

def measurement_setups(colorings, n):
	measurements = [[[i for j in range(n)] for i in 'XYZ']]
	for coloring in colorings:
		col_mes = []
		for perm in permutations('XYZ'):
			col_mes.append([perm[x] for x in coloring])
		measurements.append(col_mes)
	return measurements

def pauli_setups(mes_setups, M, n):
	mes = []
	for coloring in mes_setups:
		color_mes = []
		for col_mes in coloring:
			pauli_observable = tensor([s_dict[k] for k in col_mes])
			effects = []
			for i in range(len(pauli_observable.eigenstates()[1])):
				effects.append(pauli_observable.eigenstates()[1][i] * pauli_observable.eigenstates()[1][i].dag())
			color_mes.append(effects)
		mes.append(color_mes)
	return mes

def pauli_setups2(mes_setups, M, n):
	mes = []
	for coloring in mes_setups:
		color_mes = []
		for col_mes in coloring:
			projection_bases = [pauli_projection_dict[k] for k in col_mes]
			effects = []
			for i in range(len(M)):
				effects.append(tensor([projection_bases[k][int(M[i][k])] for k in range(n)]))
			color_mes.append(effects)
		mes.append(color_mes)
	return mes

def pauli_2qubit_setups(mes_setups, q1, q2, M2):
	mes = []
	for coloring in mes_setups:
		color_mes = []
		for col_mes in coloring:
			col_mes = [col_mes[q1], col_mes[q2]]
			projection_bases = [pauli_projection_dict[k] for k in col_mes]
			effects = []
			for i in range(len(M2)):
				effects.append(tensor([projection_bases[k][int(M2[i][k])] for k in range(2)]))
			color_mes.append(effects)
		mes.append(color_mes)
	return mes

def num_of_setups(n):
	return 6 * np.ceil(np.log(n)/np.log(3)) + 3

#Convert simulated statistics to 2-qubit statistics
#Find the coloring where the colors of q1 and q2 are different
def find_statistic(statistics, q1, q2, n):
	q_setup = None
	col_idx = 0
	setup_idx = 0
	for i in range(len(measurement_setups(colorings(n), n))):
		for j in range(len(measurement_setups(colorings(n), n)[i])):
			if measurement_setups(colorings(n), n)[i][j][q1] != measurement_setups(colorings(n), n)[i][j][q2]:
				q_setup = measurement_setups(colorings(n), n)[i]
				col_idx = i
				setup_idx = j
				break
		if q_setup:
			break
	return q_setup, col_idx, setup_idx

#Convert simulated statistics to 2-qubit statistics
def convert_statistics(statistics, q1, q2, M, M2):
	conv_stat = []
	for i in range(len(statistics)):
		s = M[statistics[i]]
		c_s = s[q1] + s[q2]
		conv_stat.append(M2.index(c_s))
	return conv_stat

#Convert simulated statistics to 2-qubit statistics
def convert_single_statistics(statistics, q1):
	conv_stat = []
	for i in range(len(statistics)):
		s = M[statistics[i]]
		c_s = s[q1]
		conv_stat.append(M1.index(c_s))
	return conv_stat

#Parametrize 2-qubit density matrices, 16 parameters
def qubit2density(p):#p is list of 16 numbers
	d = np.zeros([4, 4], dtype=complex)
	p = list(p)
	#Set diagonal elements
	for i in range(4):
		d[i][i] = p.pop(0)
	#set real elements
	for i in range(3):
		for j in range(i+1, 4):
			elem = p.pop(0)
			d[i][j] = elem
			d[j][i] = elem
	#set complex elements
	for i in range(3):
		for j in range(i+1, 4):
			elem = p.pop(0)
			d[i][j] += elem*(-1j)
			d[j][i] += elem*(1j)
	return d


def density_to_vector(density):#p is list of 16 numbers
	d = np.array(density, dtype=complex)
	p = []
	#Set diagonal elements
	for i in range(4):
		#d[i][i] = p.pop(0)
		p.append(np.real(d[i][i]))
	#set real elements
	for i in range(3):
		for j in range(i+1, 4):
			p.append(np.real(d[i][j]))
	#set complex elements
	for i in range(3):
		for j in range(i+1, 4):
			p.append(np.imag(d[i][j]))
	return tuple(p)

def maximum_likelihood(density_matrix, observable, j, k, last_index, statistics, q1, q2, M, M2):
	max_sum = 0
	stats = convert_statistics(statistics[j][k], q1, q2, M, M2)[:last_index]
	for i in range(len(observable)):
		s = np.trace(observable[i] * density_matrix)
		if s != 0:
			max_sum += stats.count(i)*np.log(np.real(s))
	return np.real(max_sum) / len(stats)

def total_maximum_likelihood(init_args, p_2obs, part_stat, last_index, stats, q1, q2, M, M2):
	max_sum = 0
	density_matrix = qubit2density(init_args)
	max_sum += maximum_likelihood(density_matrix, p_2obs[0][0], 0, 0, last_index, stats, q1, q2, M, M2)
	max_sum += maximum_likelihood(density_matrix, p_2obs[0][1], 0, 1, last_index, stats, q1, q2, M, M2)
	max_sum += maximum_likelihood(density_matrix, p_2obs[0][2], 0, 2, last_index, stats, q1, q2, M, M2)
	'''
	max_sum += maximum_likelihood(density_matrix, p_2obs[3], 
									convert_single_statistics(simulated_statistic[0][0], q2), 
									copies / num_of_setups(n))
	max_sum += maximum_likelihood(density_matrix, p_2obs[4], 
									convert_single_statistics(simulated_statistic[0][1], q2), 
									copies / num_of_setups(n))
	max_sum += maximum_likelihood(density_matrix, p_2obs[5], 
									convert_single_statistics(simulated_statistic[0][2], q2), 
									copies / num_of_setups(n))
	'''
	for i, setup in enumerate(p_2obs[part_stat[1]]):		
		max_sum += maximum_likelihood(density_matrix, p_2obs[part_stat[1]][i], part_stat[1], i, last_index, stats, q1, q2, M, M2)
	return -max_sum 

#Function for fitting
def func(x, a, b, c):
	return a - b / np.log(c * x)
 


def main(n, copies, q1, q2, w_state, start, step, state_name, num_of_runs, seed=False):

	if seed:
		random.seed(seed)

	#SIC-POVM for qubit
	E1 = rho([1,1,1])/2
	E2 = rho([1,-1,-1])/2
	E3 = rho([-1,1,-1])/2
	E4 = rho([-1,-1,1])/2

	E = [E1, E2, E3, E4]

	pauli_tensor_effects = []
	pe = tensor(s[0], s[1], s[2])
	for i in range(len(pe.eigenstates()[1])):
		pauli_tensor_effects.append(pe.eigenstates()[1][i] * pe.eigenstates()[1][i].dag())

	M = ["".join(item) for item in itertools.product("01", repeat=n)]
	M2 = ["".join(item) for item in itertools.product("01", repeat=2)]
	M1 = ["".join(item) for item in itertools.product("01", repeat=1)]

	p_obs1 = pauli_setups(measurement_setups(colorings(n), n), M, n)
	p_obs2 = pauli_setups2(measurement_setups(colorings(n), n) , M, n)

	probabilities = []
	for i in range(len(measurement_setups(colorings(n), n))):
		col_probs = []
		for j in range(len(measurement_setups(colorings(n), n)[i])):
			probs= []
			for k in range(len(p_obs2[i][j])):
				p = (w_state * p_obs2[i][j][k]).tr()
				probs.append(np.real(p))
			col_probs.append(probs)
		probabilities.append(col_probs)

	#Simulate outcome statistics
	simulated_statistic = []

	for i, coloring in enumerate(measurement_setups(colorings(n), n)):
		col_stat = []
		for j, setup in enumerate(measurement_setups(colorings(n), n)[i]):
			setup_copies = copies / num_of_setups(n)
			setup_stat = []
			for c in range(int(setup_copies)):
				rand = random.uniform(0, 1)
				cumulated_probability = 0
				for l in range(len(probabilities[i][j])):
					cumulated_probability += probabilities[i][j][l]
					if rand < cumulated_probability:
						setup_stat.append(l)
						break
			col_stat.append(setup_stat)
		simulated_statistic.append(col_stat)

	p_2obs = pauli_2qubit_setups(measurement_setups(colorings(n), n), q1, q2, M2)

	part_stat = find_statistic(simulated_statistic, q1, q2, n)

	part_sim_stat = simulated_statistic[part_stat[1]][part_stat[2]]

	last_index = len(simulated_statistic[0][0])

	'''
	p_1obs = []

	for p in 'XYZ':
		temp_obs = []
		for state in pauli_projection_dict[p]:
			temp_obs.append(tensor(state, qeye(2)))
		p_1obs.append(temp_obs)

	for p in 'XYZ':
		temp_obs = []
		for state in pauli_projection_dict[p]:
			temp_obs.append(tensor(qeye(2), state))
		p_1obs.append(temp_obs)
	'''

	bnds = ((0,1),)
	for i in range(1,16):
		if i < 4:
			bnds += ((0,1),)
		else:
			bnds += ((-1,1),)

	cons = ({'type': 'eq', 'fun': lambda x: 1 - np.trace(qubit2density(x))},
			{'type': 'ineq', 'fun': lambda x: np.real(np.linalg.eig(qubit2density(x))[0][0])},
			{'type': 'ineq', 'fun': lambda x: np.real(np.linalg.eig(qubit2density(x))[0][1])},
			{'type': 'ineq', 'fun': lambda x: np.real(np.linalg.eig(qubit2density(x))[0][2])},
			{'type': 'ineq', 'fun': lambda x: np.real(np.linalg.eig(qubit2density(x))[0][3])})

	partial_W = Qobj(np.array(w_state.ptrace([q1, q2])))

	fids = []
	k_indexes = []

	for i in range(start, len(simulated_statistic[0][0]) + 1, step):
		#print("-------------------ROUND {} of {} -----------------".format(int((i-start)/step), int(int(copies / num_of_setups(n))/step)))
		last_index = i
		#print("Last index: ------------ {}".format(last_index))
		#print(int(copies / num_of_setups(n)))
		init_quess = density_to_vector(rand_dm(4))
		reconstrution = scipy.optimize.minimize(total_maximum_likelihood, init_quess, args=(p_2obs, part_stat, 
											last_index, simulated_statistic, q1, q2, M, M2),
											bounds=bnds, constraints=cons, 
											method='SLSQP', options={'maxiter': 5000, 'disp': False})
		#if reconstrution['message'] != "Optimization terminated successfully.":
		#	print(reconstrution['message'])
			#continue
		sol_den = Qobj(qubit2density(reconstrution['x']))
		fids.append(fidelity(partial_W, sol_den))
		k_indexes.append(last_index)
		#print("------------------ FIDELITY ----------------------")
		#print(fidelity(partial_W, sol_den))
		#print(total_maximum_likelihood(density_to_vector(sol_den)))
		#print(total_maximum_likelihood(density_to_vector(partial_W)))

	k_indexes = np.array(k_indexes)
	fids = np.array(fids)

	#plt.scatter(k_indexes, fids)
	'''
	success = False
	
	while not success:
		try:
			popt, pcov = scipy.optimize.curve_fit(func, k_indexes, fids, bounds=((0, -np.inf, -np.inf), (1, np.inf, np.inf)))
			success = True
		except:
			k_indexes = k_indexes[1:]
			fids = fids[1:]
	'''
	#plt.plot(k_indexes, func(k_indexes, *popt))

	#plt.show()

	try:
		with open(r'Results\results_{}_{}_{}_{}_{}_{}_pauli.pkl'.format(n, copies, num_of_runs, q1, q2, state_name), 'rb') as f:
			results = pickle.load(f)
		results.append([k_indexes, fids])
		with open(r'Results\results_{}_{}_{}_{}_{}_{}_pauli.pkl'.format(n, copies, num_of_runs, q1, q2, state_name), 'wb') as f:
			pickle.dump(results, f)
	except:
		results = []
		results.append([k_indexes, fids])
		with open(r'Results\results_{}_{}_{}_{}_{}_{}_pauli.pkl'.format(n, copies, num_of_runs, q1, q2, state_name), 'wb') as f:
			pickle.dump(results, f)