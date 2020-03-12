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
k = 50000

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

print(colorings(4))

def measurement_setups(colorings):
	measurements = []
	for coloring in colorings:
		col_mes = []
		for perm in permutations('XYZ'):
			col_mes.append([perm[x] for x in coloring])
		measurements.append(col_mes)
	return measurements

print(measurement_setups(colorings(4)))

