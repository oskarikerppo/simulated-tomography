from qutip import *
import numpy as np
import itertools

#Number of qubits
n = 4
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
	w_vec.append(tensor(vec).unit())		
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

expectations = expect(m_obs, w_state)
print(expectations)
print(sum(expectations))