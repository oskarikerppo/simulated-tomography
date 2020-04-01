from qutip import *
import numpy as np
import scipy
import itertools
import random
import matplotlib.pyplot as plt
import pickle


#Pauli matrices
s = [sigmax(), sigmay(), sigmaz()]

#General qubit state, input as list of Bloch vector components, i.e. r = [rx, ry, rz]
def rho(r):
	if np.linalg.norm(r) != 1:
		r = np.array(r)/np.linalg.norm(r)
	return (qeye(2) + sum([r[i] * s[i] for i in range(3)])) / 2

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

#Convert simulated statistics to 2-qubit statistics
def convert_statistics(statistics, q1, q2, M, M2):
	conv_stat = []
	for i in range(len(statistics)):
		s = M[statistics[i]]
		c_s = s[q1] + s[q2]
		conv_stat.append(M2.index(c_s))
	return conv_stat


def maximum_likelihood(p, m_obs2, frequencies):
	density_matrix = qubit2density(p)
	max_sum = 0
	for i in range(len(m_obs2)):
		s = np.real(np.trace(m_obs2[i] * density_matrix))
		if s != 0:
			max_sum += frequencies.count(i)*np.log(s)
	return np.real(-max_sum/len(frequencies))

#Function for fitting
def func(x, a, b, c):
	return a - b / np.log(c * x)
 
def main(n, k, q1, q2, w_state, start, step, state_name, num_of_runs, seed=False):
	if seed:
		random.seed(seed)

	calculate_average = False
	if q1 == 0 and q2 == 0:
		calculate_average = True

	#SIC-POVM for qubit
	E1 = rho([1,1,1])/2
	E2 = rho([1,-1,-1])/2
	E3 = rho([-1,1,-1])/2
	E4 = rho([-1,-1,1])/2

	E = [E1, E2, E3, E4]

	#N-qubit POVM from the SIC-POVMs
	M = ["".join(item) for item in itertools.product("0123", repeat=n)]

	m_obs = []
	for i in range(len(M)):
		effects = []
		for j in range(n):
			effects.append(E[int(M[i][j])])
		m_obs.append(tensor(effects))

	expectations = []
	for i in range(len(m_obs)):
		expectations.append(np.real((m_obs[i] * w_state).tr()))

	#Simulate outcome statistics
	simulated_statistic = []
	for i in range(k):
		rand = random.uniform(0, 1)
		cumulated_probability = 0
		for j in range(len(expectations)):
			cumulated_probability += expectations[j]
			if rand < cumulated_probability:
				simulated_statistic.append(j)
				break

	if len(simulated_statistic) != k:
		print("Simulation of outcome statistics failed!")
		print(len(simulated_statistic))
		raise

	#Reconstruct 2-qubit states from outcome statistic

	

	#2-qubit optimization, maximum likelihood

	#2-qubit observable
	M2 = ["".join(item) for item in itertools.product("0123", repeat=2)]

	m_obs2 = []
	for i in range(len(M2)):
		effects = []
		for j in range(2):
			effects.append(E[int(M2[i][j])])
		m_obs2.append(tensor(effects))


	#Random initial guess
	args = density_to_vector(rand_dm(4))

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



	if calculate_average:
		avg_fids = []
		for q_1 in range(n):
			for q_2 in range(q_1 + 1, n):

				partial_W = Qobj(np.array(w_state.ptrace([q_1, q_2])))

				frequencies = convert_statistics(simulated_statistic, q_1, q_2, M, M2)
				true_frequencies = frequencies

				fids = []
				k_indexes = []

				for i in range(start, k, step):
					#print("-------------------ROUND {} of {} -----------------".format(int((i-start)/step), int(k/step)))
					last_index = i
					frequencies = true_frequencies[:last_index]
					init_guess = density_to_vector(rand_dm(4))
					reconstrution = scipy.optimize.minimize(maximum_likelihood, init_guess, args=(m_obs2, frequencies),
														bounds=bnds, constraints=cons, 
														method='SLSQP', options={'maxiter': 5000, 'disp': False})
					sol_den = Qobj(qubit2density(reconstrution['x']))
					fids.append(fidelity(partial_W, sol_den))
					k_indexes.append(last_index)

				k_indexes = np.array(k_indexes)
				fids = np.array(fids)
				avg_fids.append(fids)

		average_fid = np.zeros(len(avg_fids[0]))
		for i in range(len(avg_fids)):
			for j in range(len(avg_fids[i])):
				average_fid[j] += avg_fids[i][j] / len(avg_fids)

		fids = np.array(average_fid)

		try:
			with open(r'Results\results_{}_{}_{}_{}_{}_{}_sic.pkl'.format(n, k, num_of_runs, q1, q2, state_name), 'rb') as f:
				results = pickle.load(f)
			results.append([k_indexes, fids])
			with open(r'Results\results_{}_{}_{}_{}_{}_{}_sic.pkl'.format(n, k, num_of_runs, q1, q2, state_name), 'wb') as f:
				pickle.dump(results, f)
		except:
			results = []
			results.append([k_indexes, fids])
			with open(r'Results\results_{}_{}_{}_{}_{}_{}_sic.pkl'.format(n, k, num_of_runs, q1, q2, state_name), 'wb') as f:
				pickle.dump(results, f)
	else:
		partial_W = Qobj(np.array(w_state.ptrace([q1, q2])))
		frequencies = convert_statistics(simulated_statistic, q1, q2, M, M2)
		true_frequencies = frequencies

		fids = []
		k_indexes = []

		for i in range(start, k, step):
			last_index = i
			frequencies = true_frequencies[:last_index]
			init_guess = density_to_vector(rand_dm(4))
			reconstrution = scipy.optimize.minimize(maximum_likelihood, init_guess, args=(m_obs2, frequencies),
													bounds=bnds, constraints=cons, 
													method='SLSQP', options={'maxiter': 5000, 'disp': False})
			sol_den = Qobj(qubit2density(reconstrution['x']))
			fids.append(fidelity(partial_W, sol_den))
			k_indexes.append(last_index)

		k_indexes = np.array(k_indexes)
		fids = np.array(fids)

		try:
			with open(r'Results\results_{}_{}_{}_{}_{}_{}_sic.pkl'.format(n, k, num_of_runs, q1, q2, state_name), 'rb') as f:
				results = pickle.load(f)
			results.append([k_indexes, fids])
			with open(r'Results\results_{}_{}_{}_{}_{}_{}_sic.pkl'.format(n, k, num_of_runs, q1, q2, state_name), 'wb') as f:
				pickle.dump(results, f)
		except:
			results = []
			results.append([k_indexes, fids])
			with open(r'Results\results_{}_{}_{}_{}_{}_{}_sic.pkl'.format(n, k, num_of_runs, q1, q2, state_name), 'wb') as f:
				pickle.dump(results, f)