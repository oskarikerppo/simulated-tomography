from qutip import *
import numpy as np

#Number of qubits
n = 4

#W state of N qubits
w_vec = sum([basis(n, i) for i in range(n)]).unit()
print(w_vec)

#Density matrix for W state
w_state = w_vec * w_vec.dag()
print(w_state)

#Pauli matrices
s = [sigmax(), sigmay(), sigmaz()]

#General qubit state, input as list of Bloch vector components, i.e. r = [rx, ry, rz]
def rho(r):
	return (qeye(2) + sum([r[i] * s[i] for i in range(3)])) / 2
print(rho([0,1,0]))


#SIC-POVM for qubit
E1 = rho([1,1,1])/2
E2 = rho([1,-1,-1])/2
E3 = rho([-1,1,-1])/2
E4 = rho([-1,-1,1])/2

E = [E1, E2, E3, E4]
print(E)
print(sum(E))

print(tensor(E[0], E[1]))
