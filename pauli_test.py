from qutip import *
import numpy as np
import scipy
import itertools
import random
import matplotlib.pyplot as plt
from itertools import permutations


random.seed(1)


#Number of qubits
n = 4

#Number of copies
copies = 5000

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


#w_vec = sum([basis(n, i) for i in range(n)]).unit()
#print(w_vec)

#Density matrix for W state
w_state = w_vec * w_vec.dag()
#print(w_state)

#print("Eigenvalues and vectors of the W state")
#print(w_state.eigenstates())


#Pauli matrices
s = [sigmax(), sigmay(), sigmaz()]
s_dict = {'X': sigmax(), 'Y': sigmay(), 'Z': sigmaz()}

#General qubit state, input as list of Bloch vector components, i.e. r = [rx, ry, rz]
def rho(r):
	if np.linalg.norm(r) != 1:
		r = np.array(r)/np.linalg.norm(r)
	return (qeye(2) + sum([r[i] * s[i] for i in range(3)])) / 2

#print(rho([0,0,1]))


#SIC-POVM for qubit
E1 = rho([1,1,1])/2
E2 = rho([1,-1,-1])/2
E3 = rho([-1,1,-1])/2
E4 = rho([-1,-1,1])/2

E = [E1, E2, E3, E4]


print("Pauli X")
print(s[0].eigenstates()[1][0] * s[0].eigenstates()[1][0].dag())
print(s[0].eigenstates()[1][1] * s[0].eigenstates()[1][1].dag())

PX = [s[0].eigenstates()[1][0] * s[0].eigenstates()[1][0].dag(),
	  s[0].eigenstates()[1][1] * s[0].eigenstates()[1][1].dag()]

print("Pauli Y")
print(s[1].eigenstates()[1][0] * s[1].eigenstates()[1][0].dag())
print(s[1].eigenstates()[1][1] * s[1].eigenstates()[1][1].dag())

PY = [s[1].eigenstates()[1][0] * s[1].eigenstates()[1][0].dag(),
	  s[1].eigenstates()[1][1] * s[1].eigenstates()[1][1].dag()]

print("Pauli Z")
print(s[2].eigenstates()[1][0] * s[2].eigenstates()[1][0].dag())
print(s[2].eigenstates()[1][1] * s[2].eigenstates()[1][1].dag())

PZ = [s[2].eigenstates()[1][0] * s[2].eigenstates()[1][0].dag(),
	  s[2].eigenstates()[1][1] * s[2].eigenstates()[1][1].dag()]


pauli_projection_dict = {
	'X': PX, 'Y': PY, 'Z': PZ
}

print(tensor(PX[0], PX[1]))
'''
print((tensor(PX[0],PY[0],PZ[1]) * w_state).tr())



print(expect(tensor(s[0], s[0], s[2]) , w_state))

'''
pauli_tensor_effects = []
pe = tensor(s[0], s[1], s[2])
for i in range(len(pe.eigenstates()[1])):
	pauli_tensor_effects.append(pe.eigenstates()[1][i] * pe.eigenstates()[1][i].dag())
print(pauli_tensor_effects)

print(sum(pauli_tensor_effects))




def colorings(n):#n is number of qubits as int
	l = [x for x in range(int(np.ceil(np.log(n)/np.log(3))))]
	cols = []
	for c in l:
		cols.append([int(np.floor(i/(3**c)) % 3) for i in range(n)])
	return cols

print(colorings(n))

def measurement_setups(colorings):
	measurements = [[[i for j in range(n)] for i in 'XYZ']]
	for coloring in colorings:
		col_mes = []
		for perm in permutations('XYZ'):
			col_mes.append([perm[x] for x in coloring])
		measurements.append(col_mes)
	return measurements

print(measurement_setups(colorings(n)))

def pauli_setups(mes_setups):
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

			

M = ["".join(item) for item in itertools.product("01", repeat=n)]
print(M)


M2 = ["".join(item) for item in itertools.product("01", repeat=2)]
M1 = ["".join(item) for item in itertools.product("01", repeat=1)]


def pauli_setups2(mes_setups):
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

def pauli_2qubit_setups(mes_setups, q1, q2):
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
	


'''
m_obs = []
for i in range(len(M)):
	effects = []
	for j in range(n):
		effects.append(E[int(M[i][j])])
	m_obs.append(tensor(effects))
'''
p_obs = pauli_setups(measurement_setups(colorings(n)))
p_obs2 = pauli_setups2(measurement_setups(colorings(n)))
print(p_obs2 == p_obs2)




print(len(p_obs2))
print(colorings(n))

print(len(p_obs2[0]))
print(measurement_setups(colorings(n)))

print(len(p_obs2[0][0]))

#print(p_obs2[0][0][0])

f = 0
for i in range(len(p_obs2[0][0])):
	print((w_state * p_obs2[0][0][i]).tr())
	f += (w_state * p_obs2[0][0][i]).tr()
print(f)




probabilities = []
for i in range(len(measurement_setups(colorings(n)))):
	col_probs = []
	for j in range(len(measurement_setups(colorings(n))[i])):
		probs= []
		for k in range(len(p_obs2[i][j])):
			probs.append((w_state * p_obs2[i][j][k]).tr())
		col_probs.append(probs)
	probabilities.append(col_probs)
print(probabilities)


def num_of_setups(n):
	return 6 * np.ceil(np.log(n)/np.log(3)) + 3

#Simulate outcome statistics
simulated_statistic = []

for i, coloring in enumerate(measurement_setups(colorings(n))):
	col_stat = []
	for j, setup in enumerate(measurement_setups(colorings(n))[i]):
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

print(simulated_statistic)
#print(len(simulated_statistic))
#print(len(simulated_statistic[0]))
#print(len(simulated_statistic[0][0]))
#print(simulated_statistic[0][0])
			




p_2obs = pauli_2qubit_setups(measurement_setups(colorings(n)), 0, 3)
print(colorings(2))
print(measurement_setups(colorings(2)))
print(len(p_2obs[0]))
#print(p_2obs)

print("SIGMA X")
print(sigmax().eigenstates())
print("SIGMA Y")
print(sigmay().eigenstates())
print("SIGMA Z")
print(sigmaz().eigenstates())


print("SIGMA X TENSOR ID")
#print(tensor(qeye(2), sigmax()).eigenstates())
print(tensor(pauli_projection_dict['X'][0], qeye(2)).eigenstates())

print(M2)

print(simulated_statistic)

#Convert simulated statistics to 2-qubit statistics
#Find the coloring where the colors of q1 and q2 are different
def find_statistic(statistics, q1, q2):
	q_setup = None
	col_idx = 0
	setup_idx = 0
	for i in range(len(measurement_setups(colorings(n)))):
		for j in range(len(measurement_setups(colorings(n))[i])):
			if measurement_setups(colorings(n))[i][j][q1] != measurement_setups(colorings(n))[i][j][q2]:
				q_setup = measurement_setups(colorings(n))[i]
				col_idx = i
				setup_idx = j
				break
		if q_setup:
			break
	return q_setup, col_idx, setup_idx





print(measurement_setups(colorings(n)))

print(find_statistic(simulated_statistic, 0, 3))

part_stat = find_statistic(simulated_statistic, 0, 3)

print(simulated_statistic[part_stat[1]][part_stat[2]])

part_sim_stat = simulated_statistic[part_stat[1]][part_stat[2]]


#Convert simulated statistics to 2-qubit statistics
def convert_statistics(statistics, q1, q2):
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


print(convert_statistics(part_sim_stat, 0, 3))
print(M)
print(M1)
print(M2)
print(convert_single_statistics(part_sim_stat, 0))
print(convert_single_statistics(part_sim_stat, 1))

print(part_stat[1])

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

def maximum_likelihood(p, observable, freqs, cops):
	density_matrix = qubit2density(p)
	max_sum = 0
	for i in range(len(observable)):
		max_sum += freqs.count(i)*np.log(np.real(np.trace(observable[i] * density_matrix)))
	return np.real(-max_sum / cops)


print(part_stat)
#print(p_obs2[part_stat[1]][0])
args = density_to_vector(rand_dm(4))
print(len(p_2obs[1][0]))
x = maximum_likelihood(args, p_2obs[part_stat[1]][1], 
						convert_statistics(simulated_statistic[1][1], 0, 3), 
						len(convert_statistics(simulated_statistic[1][1], 0, 3)))

print(x)


#print(tensor(qeye(2), sigmax()).eigenstates()[1])

#for state in tensor(qeye(2), sigmax()).eigenstates()[1]:
#	print(state*state.dag())

#print("------------------------------")

for state in pauli_projection_dict['X']:
	print(tensor(qeye(2), state))


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




args = density_to_vector(rand_dm(4))

def total_maximum_likelihood(args):
	max_sum = 0
	'''
	max_sum += maximum_likelihood(args, p_1obs[0], 
									convert_single_statistics(simulated_statistic[0][0], 0), 
									copies / num_of_setups(n))
	max_sum += maximum_likelihood(args, p_1obs[1], 
									convert_single_statistics(simulated_statistic[0][1], 0), 
									copies / num_of_setups(n))
	max_sum += maximum_likelihood(args, p_1obs[2], 
									convert_single_statistics(simulated_statistic[0][2], 0), 
									copies / num_of_setups(n))
	max_sum += maximum_likelihood(args, p_1obs[3], 
									convert_single_statistics(simulated_statistic[0][0], 1), 
									copies / num_of_setups(n))
	max_sum += maximum_likelihood(args, p_1obs[4], 
									convert_single_statistics(simulated_statistic[0][1], 1), 
									copies / num_of_setups(n))
	max_sum += maximum_likelihood(args, p_1obs[5], 
									convert_single_statistics(simulated_statistic[0][2], 1), 
									copies / num_of_setups(n))
	'''
	for i, setup in enumerate(p_obs2[part_stat[1]]):		
		max_sum += maximum_likelihood(args, p_2obs[part_stat[1]][i], 
						convert_statistics(simulated_statistic[part_stat[1]][i], 0, 3), 
						len(convert_statistics(simulated_statistic[part_stat[1]][i], 0, 3)))



	return max_sum



print(total_maximum_likelihood(args))


bnds = ((0,1),)
for i in range(1,16):
	if i < 4:
		bnds += ((0,1),)
	else:
		bnds += ((-1,1),)

#print(bnds)

cons = ({'type': 'eq', 'fun': lambda x: 1 - np.trace(qubit2density(x))},
		{'type': 'ineq', 'fun': lambda x: np.real(np.linalg.eig(qubit2density(x))[0][0])},
		{'type': 'ineq', 'fun': lambda x: np.real(np.linalg.eig(qubit2density(x))[0][1])},
		{'type': 'ineq', 'fun': lambda x: np.real(np.linalg.eig(qubit2density(x))[0][2])},
		{'type': 'ineq', 'fun': lambda x: np.real(np.linalg.eig(qubit2density(x))[0][3])})








reconstrution = scipy.optimize.minimize(total_maximum_likelihood, args, 
										bounds=bnds, constraints=cons, 
										method='SLSQP', options={'maxiter': 250, 'disp': True})
print(reconstrution)

sol_den = Qobj(qubit2density(reconstrution['x']))

partial_W = Qobj(np.array(w_state.ptrace([0,3])))


print("-----------------RECONSTRUCTED------------")
print(sol_den)
print(sol_den.eigenstates())

print("-----------------PARTIAL W-----------------")
print(partial_W)
print(partial_W.eigenstates())


print(fidelity(partial_W, sol_den))

print(sol_den.tr())



true_statistics = simulated_statistic

fids = []
k_indexes = []

res = 10
'''
for i in range(5, int(k / res), int(k/(res*25))):
	print("-------------------PRELIMINARY ROUND {} of {} -----------------".format(i, int(k / res)))
	last_index = i
	frequencies = true_frequencies[:last_index]
	args = density_to_vector(rand_dm(4))
	reconstrution = scipy.optimize.minimize(maximum_likelihood, args, 
										bounds=bnds, constraints=cons, 
										method='SLSQP')
	sol_den = Qobj(qubit2density(reconstrution['x']))
	fids.append(fidelity(partial_W, sol_den))
	k_indexes.append(last_index)
'''

step = 10
start = 5
for i in range(start, int(copies / num_of_setups(n)), step):
	print("-------------------ROUND {} of {} -----------------".format(int((i-start)/step), int(int(copies / num_of_setups(n))/step)))
	last_index = i
	for i in range(len(simulated_statistic)):
		for j in range(len(simulated_statistic[i])):
			simulated_statistic[i][j] = true_statistics[i][j][:last_index]
	args = density_to_vector(rand_dm(4))
	reconstrution = scipy.optimize.minimize(total_maximum_likelihood, args, 
										bounds=bnds, constraints=cons, 
										method='SLSQP', options={'maxiter': 250, 'disp': False})
	sol_den = Qobj(qubit2density(reconstrution['x']))
	fids.append(fidelity(partial_W, sol_den))
	k_indexes.append(last_index)

k_indexes = np.array(k_indexes)
fids = np.array(fids)


plt.scatter(k_indexes, fids)

#Function for fitting
def func(x, a, b, c):
	return a - b / np.log(c * x)
 
#popt, pcov = scipy.optimize.curve_fit(func, k_indexes, fids, bounds=((0, -np.inf, -np.inf), (1, np.inf, np.inf)))
#print(popt)
#print(pcov)
#fit = np.poly1d(np.polyfit(k_indexes, np.log(fids), 1, w=np.sqrt(fids)))



#plt.plot(k_indexes, func(k_indexes, *popt))

plt.show()