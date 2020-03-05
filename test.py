from qutip import *
import numpy as np
import scipy
import itertools
import random


random.seed(1)


#Number of qubits
n = 3

#Number of copies
k = 50

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

print("Eigenvalues and vectors of the W state")
print(w_state.eigenstates())


#Pauli matrices
s = [sigmax(), sigmay(), sigmaz()]

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




#print(E)
#print(sum(E))

#print(tensor(E[0], E[1]))

#N-qubit POVM from the SIC-POVMs
M = ["".join(item) for item in itertools.product("0123", repeat=n)]
#print(M)
#print(len(M))
m_obs = []
for i in range(len(M)):
	effects = []
	for j in range(n):
		effects.append(E[int(M[i][j])])
	m_obs.append(tensor(effects))
#print(m_obs)
print(sum(m_obs))
print(len(m_obs))

expectations = []
for i in range(len(m_obs)):
	expectations.append(np.real((m_obs[i] * w_state).tr()))
#print(expectations)
print(sum(expectations))



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

print(simulated_statistic)

if len(simulated_statistic) != k:
	print("Simulation of outcome statistics failed!")
	print(len(simulated_statistic))
	raise


print(M[simulated_statistic[0]])



#Reconstruct 2-qubit states from outcome statistic

partial_W = w_state.ptrace([0, 2])
#print(w_state)
#print(partial_W)


'''
Uxy = [[(m_obs[i] * m_obs[j]).tr() for j in range(len(m_obs))] for i in range(len(m_obs))]

Uxy_inv = np.linalg.inv(Uxy)

r_vec = []
for i in range(len(m_obs)):
	temp_sum = 0
	for j in range(len(m_obs)):
		temp_sum += np.real(Uxy_inv[i][j]*simulated_statistic.count(j) / k)
	r_vec.append(temp_sum)

print(r_vec)
reconstructed_state = sum([r_vec[i] * m_obs[i] for i in range(len(m_obs))])
print(reconstructed_state)

print("----------------- RECONSTRUCTED EIGENVALUES -----------------------")

print(reconstructed_state.eigenstates())


#Trace distance of original W state and reconsturcted n-qubit state
fid = fidelity(w_state, reconstructed_state)
print(fid)

print(reconstructed_state.tr())

'''

#2-qubit optimization, maximum likelihood

#2-qubit observable
M2 = ["".join(item) for item in itertools.product("0123", repeat=2)]
#print(M2)
#print(len(M))
m_obs2 = []
for i in range(len(M2)):
	effects = []
	for j in range(2):
		effects.append(E[int(M2[i][j])])
	m_obs2.append(tensor(effects))
#print(m_obs)
#print(sum(m_obs2))
#print(len(m_obs2))
print("--------------------------EIGENVALUES----------------------------")
print(np.linalg.eig(partial_W)[0][0])

#print(partial_W.eigenstates())


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

#print(qubit2density([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]))


#Convert simulated statistics to 2-qubit statistics
def convert_statistics(statistics, q1, q2):
	conv_stat = []
	for i in range(len(statistics)):
		s = M[statistics[i]]
		c_s = s[q1] + s[q2]
		conv_stat.append(M2.index(c_s))
	return conv_stat

print(simulated_statistic)
print(convert_statistics(simulated_statistic, 0, 1))

frequencies = convert_statistics(simulated_statistic, 0, 1)

def maximum_likelihood(density_matrix):
	max_sum = 0
	for i in range(len(m_obs2)):
		max_sum += frequencies[i]*np.log(np.real((m_obs2[i] * density_matrix).tr())) 
	return -max_sum / k



print(maximum_likelihood(partial_W))

args = list(np.random.uniform(0, 1, 4)) + list(np.random.uniform(-1, 1, 12))
#print(args)

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


reconstrution = scipy.optimize.minimize(maximum_likelihood, args, bounds=bnds,
			 constraints=cons)
print(reconstrution)