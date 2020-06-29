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




haar_ket = rand_ket_haar(N=3, dims=None, seed=None)

haar_dm = haar_ket * haar_ket.dag()

def state_0000():
	one = tensor([basis(2,0) for i in range(4)])
	zero = tensor([basis(2,1) for i in range(4)])
	dm = (zero * zero.dag() + one * one.dag()) / 2
	return dm


def state_0011():
	s_00 = tensor([basis(2,1), basis(2,1)])
	s_01 = tensor([basis(2,1), basis(2,0)])
	s_10 = tensor([basis(2,0), basis(2,1)])
	s_11 = tensor([basis(2,0), basis(2,0)])
	s = [s_00, s_01, s_10, s_11]
	dm = sum([tensor(x * x.dag(), x * x.dag()) for x in s]) / 4
	return dm

print(state_0000().shape)
print(state_0000().dims)

print(state_0011().shape)
print(state_0011().dims)

print(Qobj(state_0000().data).shape)
print(Qobj(state_0000().data).dims)



def random_state(A=2, B=2, rank_lmt=2):
	a_ket = rand_ket_haar(N=2**A, dims=None, seed=None)
	b_ket = rand_ket_haar(N=2**B, dims=None, seed=None)

	a_dm = a_ket * a_ket.dag()
	b_dm = b_ket * b_ket.dag()

	dm = tensor(a_dm, b_dm)
	dms = [dm]
	'''
	print(dm)
	print(np.linalg.eig(dm)[0])
	print(len(np.linalg.eig(dm)[0]))
	print(np.linalg.matrix_rank(dm))
	print(2**A * 2**B)
	'''
	total_dim = 2**A * 2**B
	while np.linalg.matrix_rank(sum(dms)) < rank_lmt and np.linalg.matrix_rank(sum(dms)) <= total_dim:
		a_ket = rand_ket_haar(N=2**A, dims=None, seed=None)
		b_ket = rand_ket_haar(N=2**B, dims=None, seed=None)
		a_dm = a_ket * a_ket.dag()
		b_dm = b_ket * b_ket.dag()

		dm = tensor(a_dm, b_dm)
		dms.append(dm)

	print(len(dms))

	convex_weights = np.random.normal(size=len(dms))
	convex_weights = convex_weights / np.linalg.norm(convex_weights)
	convex_weights = np.array([x**2 for x in convex_weights])
	#print(convex_weights)
	#print(np.linalg.norm(convex_weights))
	#print(sum(convex_weights))
	total_dm = sum([convex_weights[i] * dms[i] for i in range(len(dms))])
	#print(total_dm)
	#print(total_dm.tr())
	#print(np.linalg.matrix_rank(total_dm))
	#print(total_dm.dims[0])
	return total_dm


def calculate_expectation(POVM, input_state, outcomes, i, n):
	effects = [POVM[int(outcomes[i][j])] for j in range(n)]
	return np.real((Qobj(np.array(tensor(effects))) * input_state).tr())

def one_to_three(key):
	indices = [0, 1, 2, 3]
	transposed = [indices.pop(key.find('1'))]
	return transposed, indices


def two_to_two(key):
	indices = [0, 1, 2, 3]
	transposed = [indices[int(i)] for i in range(4) if key[i] == '1']
	original = [indices[int(i)] for i in range(4) if key[i] == '0']
	return transposed, original


'''
print(total_dm)
print(np.linalg.eig(total_dm)[0])
print(np.linalg.matrix_rank(total_dm))
print(total_dm.tr())
'''
#print(sum([np.linalg.matrix_rank(x) for x in dms]))
#print(len(dms))

#print(x)
'''
y = tensor(basis(2,0),basis(2,0))
y = y * y.dag()
y1 = tensor(basis(2,1),basis(2,1)).unit()
y1 = y1 * y1.dag()
yy = (y + y1) / 2
print(yy)
print(yy.dims[0])
yT = partial_transpose(yy, [1, 0])
eigs = [np.real(x) for x in np.linalg.eig(yT)[0]]
print(eigs)
print(concurrence(yy))
'''

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

#init_state = w_state(8)
init_state = random_state(1,2)
print(init_state)
print(init_state.tr())
print(np.linalg.eig(init_state)[0])
print(init_state.dims)

print("---------------")
print(init_state.shape)
print(init_state.dims)




init_state = Qobj(np.array(init_state))
print(init_state)
print(init_state.tr())
print(np.linalg.eig(init_state)[0])
print(init_state.dims)


init_state1 = tensor(basis(2,0), basis(2,0), basis(2,0))
init_state1 = init_state1 * init_state1.dag()

init_state2 = tensor(basis(2,1), basis(2,1), basis(2,1))
init_state2 = init_state2 * init_state2.dag()

input_state = (init_state1 + init_state2)/2
input_state = Qobj(input_state.data)

s0= time()
print("Initializing")
povm_string = ""
for i in range(len(E)):
	povm_string += str(i)

outcomes = []
for i, item in enumerate(itertools.product(povm_string, repeat=4)):
	outcomes.append("".join(item))

#expectations = np.array([calculate_expectation(E, init_state, outcomes, i, 4) for i in range(len(outcomes))])
s1 = time()
print(s1 - s0)

fid_res = {}
neg_res = {}
ent_res = {}

try:
	with open(r'Results\Negativity\fid_res.pkl', 'rb') as f:
		fid_res = pickle.load(f)
	with open(r'Results\Negativity\neg_res.pkl', 'rb') as f:
		neg_res = pickle.load(f)
	with open(r'Results\Negativity\ent_res.pkl', 'rb') as f:
		ent_res = pickle.load(f)
except:
	pass


max_rank = 2
if  not fid_res:
	#for rank in range(1, max_rank):
	init_state = state_0011()
	fids = []
	negativities = {'1000': [], '0100': [], '0010': [], '0001': [],
					'1100': [], '1010': [], '1001': [], '0110': [], '0101': [], '0011': []}
	entropies = {'1000': [], '0100': [], '0010': [], '0001': [],
					'1100': [], '1010': [], '1001': [], '0110': [], '0101': [], '0011': []}
	rounds = 2000
	times = [time()]
	spent_times = []
	for i in range(rounds):
		print("Round {} of {}".format(i, rounds))
		init_state = Qobj(init_state.data)
		expectations = np.array([calculate_expectation(E, init_state, outcomes, i, 4) for i in range(len(outcomes))])
		res = simulate_povm.main(4, 8192*20, (0, 1, 2, 3), init_state, E, expectations, "W", "sic", seed=False)
		fids.append(res[0])

		total_dm = Qobj(res[1], dims=[[2, 2, 2, 2], [2, 2, 2, 2]], shape=[16, 16])
		original_dm = Qobj(res[2], dims=[[2, 2, 2, 2], [2, 2, 2, 2]], shape=[16, 16])

		#1-to-3
		for key in ['1000', '0100', '0010', '0001']:
			transpose_dm = partial_transpose(total_dm, [int(x) for x in key])
			eigs = np.linalg.eig(transpose_dm)[0]
			eigs = [np.real(x) for x in eigs]
			negativity = sum([abs(x) for x in eigs if x < 0])
			negativities[key].append(negativity)
			partition = one_to_three(key)
			entropies[key].append(entropy_mutual(total_dm, partition[0], partition[1], base=2))

		#2-to-2
		for key in ['1100', '1010', '1001', '0110', '0101', '0011']:
			transpose_dm = partial_transpose(total_dm, [int(x) for x in key])
			eigs = np.linalg.eig(transpose_dm)[0]
			eigs = [np.real(x) for x in eigs]
			negativity = sum([abs(x) for x in eigs if x < 0])
			negativities[key].append(negativity)
			partition = two_to_two(key)
			entropies[key].append(entropy_mutual(total_dm, partition[0], partition[1], base=2))

		#time left
		times.append(time())
		spent_times.append((times[-1] - times[-2]) / 60)
		print("Time since start: {} minutes".format((times[-1] - times[0]) / 60 ))
		print("Time this round: {} minutes".format((times[-1] - times[-2]) / 60))
		print("Estimated time left: {} hours".format(np.average(spent_times) * (rounds - i + 1) / 60))


	fid_res['0011state'] = fids
	neg_res['0011state'] = negativities
	ent_res['0011state'] = entropies
	with open(r'Results\Negativity\fid_res.pkl', 'wb') as f:
		pickle.dump(fid_res, f)
	with open(r'Results\Negativity\neg_res.pkl', 'wb') as f:
		pickle.dump(neg_res, f)
	with open(r'Results\Negativity\ent_res.pkl', 'wb') as f:
		pickle.dump(ent_res, f)


average_negs_31 = np.average(neg_res['0011state']['1000'])
average_negs_22 = np.average(neg_res['0011state']['1100'])
average_negs = [average_negs_31, average_negs_22]

std_negs_31 = np.std(neg_res['0011state']['1000'])
std_negs_22 = np.std(neg_res['0011state']['1100'])
std_negs = [std_negs_31, std_negs_22]

sem_negs_31 = stats.sem(neg_res['0011state']['1000'])
sem_negs_22 = stats.sem(neg_res['0011state']['1100'])

average_ents_31 = np.average(ent_res['0011state']['1000'])
average_ents_22 = np.average(ent_res['0011state']['1100'])
average_ents = [average_ents_31, average_ents_22]

std_ents_31 = np.std(ent_res['0011state']['1000'])
std_ents_22 = np.std(ent_res['0011state']['1100'])
std_ents = [std_ents_31, std_ents_22]

sem_ents_31 = stats.sem(ent_res['0011state']['1000'])
sem_ents_22 = stats.sem(ent_res['0011state']['1100'])

average_fids = [np.average(fid_res['0011state'])]
std_fids = [np.std(fid_res['0011state'])]
sem_fids = [stats.sem(fid_res['0011state'])]

data = [average_negs, average_ents]
xlabels = ['3-to-1', '2-to-2']


# create plot
fig, ax = plt.subplots()
index = np.arange(2)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, data[0], bar_width,
yerr=std_negs,
alpha=opacity,
color='b',
label='Negativity')

rects2 = plt.bar(index + bar_width, data[1], bar_width,
yerr=std_ents,	
alpha=opacity,
color='g',
label='Mutual entropy')

plt.xlabel('Matrix rank')
plt.ylabel('Entropy/Negativity')
plt.title('Negativity/Entropy of random states of given rank')
plt.xticks(index + bar_width/2, tuple(xlabels))
plt.legend()

plt.tight_layout()
plt.show()


for i in range(2):
	keys = ['1000', '1100']
	key = keys[i]
	plt.scatter(ent_res['0011state'][key], neg_res['0011state'][key])
	plt.xlabel("Entropy")
	plt.ylabel("Negativity")
	plt.title("Entropy vs negativity, {}".format(xlabels[i]))
	plt.show()

print(average_negs)
print(std_negs)

print(average_ents)
print(std_ents)

print(average_fids)
print(std_fids)


plt.hist(neg_res['0011state']['0101'], bins=20)
plt.show()


#Do it for 4-qubit states: 1-to-3 and 2-to-2
#Do it for 1/2 (|0000><0000| + |1111><1111|)
#Do it for 1/4 sum_b2 |b2><b2| \otimes |b2><b2|
#b2 = 00, 01, 10, 11


'''
print(np.average(negativities))
print(np.average(entropies))
print(np.std(negativities))
print(np.std(entropies))
print(np.average(fids))
print(np.std(fids))

print(fids)
print(negativities)
print(entropies)

plt.scatter(entropies, negativities)
plt.show()


print(fids)

total_dm = Qobj(fids[0][1], dims=[[2, 4], [2, 4]], shape=[8, 8])
original_dm = Qobj(fids[1][1], dims=[[2, 4], [2, 4]], shape=[8, 8])

transpose_dm = partial_transpose(total_dm, [1,0])
eigs = np.linalg.eig(transpose_dm)[0]
print(type(eigs[0]))
print(eigs)
eigs = [np.real(x) for x in eigs]
negativity = sum([abs(x) for x in eigs if x < 0])
print(negativity)
print(entropy_mutual(total_dm, 0, 1))
'''