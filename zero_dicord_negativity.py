from qutip import *
import numpy as np
import scipy
from scipy import stats
import itertools
import random
import matplotlib.pyplot as plt
import pickle
from time import time
import simulate_povm
from pathlib import Path
import os
import qdiscord

from plot_settings import *

def state_00_11(p0, p1):
	s0 = basis(2, 0) * basis(2, 0).dag()
	s1 = basis(2, 1) * basis(2, 1).dag()
	k00 = tensor([basis(2, 0), basis(2, 0)])
	s00 = k00 * k00.dag()
	k01 = tensor([basis(2, 0), basis(2, 1)])
	s01 = k01 * k01.dag()
	
	dm = p0 * tensor(s0, s00) + p1 * tensor(s1, s01)

	return dm

def state_0_1():
	s0 = basis(2, 0) * basis(2, 0).dag()
	s1 = basis(2, 1) * basis(2, 1).dag()
	theta = np.pi * random.random() / 2
	dm =  (np.cos(theta)**2) * tensor(s0, s0) + (np.sin(theta)**2) * tensor(s1, s1)

	return dm


def random_state(r):
	#return rand_dm_ginibre(4, rank=r, dims=[[2, 2], [2, 2]])
	#return rand_dm(4, dims=[[2, 2], [2, 2]])

	x = np.random.random_sample(size = r)
	#print(x)
	y = [xi/np.linalg.norm(x) for xi in x]
	#print(y)
	#print(sum([yi**2 for yi in y]))
	states = []
	for i in range(r):
		r1 = rand_dm_ginibre(2, rank=1)
		r2 = rand_dm_ginibre(2, rank=1)
		states.append(tensor(r1, r2))


	rho = sum([y[i]**2 * states[i] for i in range(r)])
	return rho


def calculate_expectation(POVM, input_state, outcomes, i, n):
	effects = [POVM[int(outcomes[i][j])] for j in range(n)]
	return np.real((Qobj(np.array(tensor(effects))) * input_state).tr())

def one_to_three(key, indices):
	transposed = [indices.pop(key.find('1'))]
	return transposed, indices

E1 = basis(2, 0) * basis(2, 0).dag() / 2
#E1 = np.array(E1)
e2 = basis(2, 0) / np.sqrt(3) + np.sqrt(2/3) * basis(2, 1)
E2 = e2 * e2.dag() / 2
#E2 = np.array(E2)
e3 = basis(2, 0) / np.sqrt(3) + np.sqrt(2/3) * np.exp(1j*2*np.pi/3) * basis(2, 1)
E3 = e3 * e3.dag() / 2
#E3 = np.array(E3)
e4 = basis(2, 0) / np.sqrt(3) + np.sqrt(2/3) * np.exp(1j*4*np.pi/3) * basis(2, 1)
E4 = e4 * e4.dag() / 2
#E4 = np.array(E4)

E = [E1, E2, E3, E4]

povm_string = ""
for i in range(len(E)):
	povm_string += str(i)

two_outcomes = []
for i, item in enumerate(itertools.product(povm_string, repeat=2)):
	two_outcomes.append("".join(item))



theta = np.pi * random.random() / 2
#print(theta)
#print(np.cos(theta), np.sin(theta))


fids = []
negs = []
ents = []
discs = []


partition = one_to_three('10', [0, 1])
#print(partition)


try:
	working_directory = Path(os.getcwd())
	print(working_directory)
	data_folder = working_directory / "Results/Negativity/"
	res_file = data_folder / "zero_dicord_neg.pkl"
	with open(res_file, 'rb') as f:
		result_file = pickle.load(f)
		ents = result_file[0]
		negs = result_file[1]
		fids = result_file[2]
		discs = result_file[3]
except:
	pass

append_new_results = True

rounds = 1000

rank = 4

theta = np.pi * random.random() / 2

#rho = state_00_11(np.cos(theta)**2, np.sin(theta)**2)
#rho = Qobj(rho.data)

#print(rho.dims)




test_rho = random_state(rank)
print(test_rho.ptrace(0))


test_rho_data = np.array(test_rho)
print(test_rho_data)

discord = qdiscord.QDiscord()
dis = np.real(discord.discord(test_rho_data))
print(dis)



if append_new_results:
	for i in range(rounds):
		print(i)
		#theta = np.pi * random.random() / 2
		#rho = state_00_11(np.cos(theta)**2, np.sin(theta)**2)
		#rho = Qobj(rho.data)

		#rho = random_state(rank)
		rho = state_0_1()
		rho = Qobj(np.array(rho))
		two_ecpectations = np.array([calculate_expectation(E, rho, two_outcomes, i, 2) for i in range(len(two_outcomes))])
		res = simulate_povm.main(2, 8192*20, (0, 1), rho, E, two_ecpectations, "W", "sic", seed=False)

		fids.append(res[0])
		total_dm = Qobj(res[1], dims=[[2, 2], [2, 2]], shape=[4, 4])
		original_dm = Qobj(res[2], dims=[[2, 2], [2, 2]], shape=[4, 4])

		transpose_dm = partial_transpose(total_dm, [1, 0])
		eigs = np.linalg.eig(transpose_dm)[0]
		eigs = [np.real(x) for x in eigs]
		negativity = sum([abs(x) for x in eigs if x < 0])
		negs.append(negativity)

		ents.append(entropy_mutual(original_dm, partition[0], partition[1], base=2))
		discs.append(np.real(discord.discord(np.array(rho))))

print(len(ents))
plt.figure(figsize=fig_size)
plt.scatter(ents, discs, s=10)
plt.xlabel("Mutual entropy")
plt.ylabel("Discord")
#plt.title("Entropy vs negativity, {}".format(xlabels[i]))
plt.show()
#plt.savefig('Figures/Negativity results/entropy-vs-negativity-zero-discord.pdf', format='pdf', bbox_inches='tight')

#print(len(ents))
plt.figure(figsize=fig_size)
plt.scatter(ents, negs, s=10)
plt.xlabel("Mutual entropy")
plt.ylabel("Negativity")
#plt.title("Entropy vs negativity, {}".format(xlabels[i]))
plt.show()
#plt.savefig('Figures/Negativity results/entropy-vs-negativity-zero-discord.pdf', format='pdf', bbox_inches='tight')

plt.figure(figsize=fig_size)
plt.scatter(negs, discs, s=10)
plt.xlabel("Negativity")
plt.ylabel("Discord")
#plt.title("Entropy vs negativity, {}".format(xlabels[i]))
plt.show()
#plt.savefig('Figures/Negativity results/entropy-vs-negativity-zero-discord.pdf', format='pdf', bbox_inches='tight')

with open(r'Results/Negativity/zero_dicord_neg.pkl', 'wb') as f:
	pickle.dump([ents, negs, fids, discs], f)
