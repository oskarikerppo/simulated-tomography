from qutip import *
import numpy as np
import scipy
from scipy import stats
import itertools
import random
import matplotlib.pyplot as plt
import pickle
from time import time

#Pauli matrices
s = [sigmax(), sigmay(), sigmaz()]

#General qubit state, input as list of Bloch vector components, i.e. r = [rx, ry, rz]
def rho(r):
	if np.linalg.norm(r) != 1:
		r = np.array(r)/np.linalg.norm(r)
	return np.array(np.array((qeye(2) + sum([r[i] * s[i] for i in range(3)])) / 2))


def random_qubit():
	theta = 2 * np.pi * np.random.rand()
	phi = np.arccos(2 * np.random.rand() - 1)
	x = np.cos(theta) * np.sin(phi)
	y = np.sin(theta) * np.sin(phi)
	z = np.cos(phi)
	return rho([x, y, z])

def qubit_fidelity(rho1, rho2):
	return np.real(np.trace(rho1 @ rho2) + 2 * np.sqrt(np.linalg.det(rho1) * np.linalg.det(rho2)))

def general_difelity(rho1, rho2):
	rho1sqrt = scipy.linalg.sqrtm(rho1)
	sqrtmat = scipy.linalg.sqrtm(rho1sqrt @ rho2 @ rho1sqrt)
	return np.real(np.trace(sqrtmat)) **2


def random_state(A=2, rank_lmt=2):
	a_ket = rand_ket(N=2**A, density=1, dims=None, seed=None)

	a_dm = a_ket * a_ket.dag()

	dms = [a_dm]

	total_dim = 2**A
	for i in range(rank_lmt):
		a_ket = rand_ket(N=2**A, density=1, dims=None, seed=None)
		#print(a_ket)
		#print(np.linalg.norm(a_ket))
		#die
		a_dm = np.array(a_ket.data @ np.conj(a_ket.data).T)
		dms.append(a_dm)

	convex_weights = np.random.normal(size=len(dms))
	convex_weights = convex_weights / np.linalg.norm(convex_weights)
	convex_weights = np.array([x**2 for x in convex_weights])

	total_dm = sum([convex_weights[i] * dms[i] for i in range(len(dms))])

	return np.array(total_dm)

#00 + 11
bell_1 = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1)))/np.sqrt(2)
#00 - 11
bell_2 = (tensor(basis(2, 0), basis(2, 0)) - tensor(basis(2, 1), basis(2, 1)))/np.sqrt(2)
#01 + 10
bell_3 = (tensor(basis(2, 0), basis(2, 1)) + tensor(basis(2, 1), basis(2, 0)))/np.sqrt(2)
#01 - 10
bell_4 = (tensor(basis(2, 0), basis(2, 1)) - tensor(basis(2, 1), basis(2, 0)))/np.sqrt(2)

bell_1 = bell_1 * bell_1.dag()
bell_2 = bell_2 * bell_2.dag()
bell_3 = bell_3 * bell_3.dag()
bell_4 = bell_4 * bell_4.dag()
bell = [bell_1, bell_2, bell_3, bell_4]

smolin = sum([tensor([bell[i], bell[i]]) for i in range(4)])/4
smolin = Qobj(smolin)
smolin = Qobj(smolin.data)
#print(np.linalg.matrix_rank(smolin))
print(smolin.dims)
print(smolin.shape)




A = 4
rank_lmt = 16
sample_size = 10000

init_state = smolin
#init_state = random_state(A, rank_lmt)
#init_state = random_qubit()
#init_state = rho([0,0,-1])

fids = []
fids2 = []
for i in range(sample_size):
	rand_state = rand_dm_ginibre(N=16, rank=4)
	#rand_state = random_state(A, rank_lmt)
	#rand_state = random_qubit()
	#fids.append(general_difelity(init_state, rand_state))
	fids2.append(fidelity(init_state, Qobj(rand_state)) ** 2 )

#fids = np.array([fidelity(init_state, random_state(A, rank_lmt)) for x in range(sample_size)])

#print(np.average(fids))
#print(np.std(fids))

print(np.average(fids2))
print(np.std(fids2))

#plt.hist(fids, 50)
#plt.show()

plt.hist(fids2, 50)
plt.show()

#print(init_state)
#print(general_difelity(init_state, init_state))



