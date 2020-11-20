import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import os
from scipy import stats
from plot_settings import *


"""
Loads data files and produces plots.
Matplotlib settings are imported from plot_settings
"""

try:
	working_directory = Path(os.getcwd())
	print(working_directory)
	data_folder = working_directory / "Results/Negativity/"
	fid_file = data_folder / "fid_res.pkl"
	neg_file = data_folder / "neg_res.pkl"
	ent_file = data_folder / "ent_res.pkl"
	with open(fid_file, 'rb') as f:
		fid_res = pickle.load(f)
	with open(neg_file, 'rb') as f:
		neg_res = pickle.load(f)
	with open(ent_file, 'rb') as f:
		ent_res = pickle.load(f)
except:
	print("Loading of data failed")
	raise

"""
Load data for 4-qubit states according to partition 1000 or 1100
Note that not all data is used to produce the plots. It is there, however, if one wished to use it.
"""

#Average negativities (fake entanglement)
average_negs_31 = np.average(neg_res['0011state']['1000'])
average_negs_22 = np.average(neg_res['0011state']['1100'])
average_negs = [average_negs_31, average_negs_22]

#Standard deviation of fake entanglement
std_negs_31 = np.std(neg_res['0011state']['1000'])
std_negs_22 = np.std(neg_res['0011state']['1100'])
std_negs = [std_negs_31, std_negs_22]

#Standard error of the mean
sem_negs_31 = stats.sem(neg_res['0011state']['1000'])
sem_negs_22 = stats.sem(neg_res['0011state']['1100'])
sem_negs = [sem_negs_31, sem_negs_22]

#Average mutual entropy (qutip entropy_mutual)
average_ents_31 = np.average(ent_res['0011state']['1000'])
average_ents_22 = np.average(ent_res['0011state']['1100'])
average_ents = [average_ents_31, average_ents_22]

#Standard deviation of mutual entropy
std_ents_31 = np.std(ent_res['0011state']['1000'])
std_ents_22 = np.std(ent_res['0011state']['1100'])
std_ents = [std_ents_31, std_ents_22]

#Standard error of the mean mutual entropy
sem_ents_31 = stats.sem(ent_res['0011state']['1000'])
sem_ents_22 = stats.sem(ent_res['0011state']['1100'])
sem_ents = [sem_ents_31, sem_ents_22]

#Average fidelities (reconstructed state vs original)
average_fids = [np.average(fid_res['0011state'])]
std_fids = [np.std(fid_res['0011state'])]
sem_fids = [stats.sem(fid_res['0011state'])]

#Data for first figure
data = [average_negs, average_ents]
xlabels = ['3-to-1', '2-to-2']


# Entropy and (fake) negativity of 4-qubit state
fig, ax = plt.subplots(figsize=fig_size)
index = np.arange(2)

#Rectangles for average negativities with standard deviation
rects1 = plt.bar(index, data[0], bar_width,
yerr=std_negs,
alpha=opacity,
color='b',
label='Negativity')

#Rectangles for average mutual entropy with standard deviation
rects2 = plt.bar(index + bar_width, data[1], bar_width,
yerr=std_ents,	
alpha=opacity,
color='g',
label='Mutual entropy')


plt.ylabel('Entropy/Negativity')
#plt.title('Negativity/Entropy of 4-qubit state')
plt.xticks(index + bar_width/2, tuple(xlabels))
plt.legend()
plt.tight_layout()
plt.show()
#fig.savefig('Figures/Negativity results/4-qubit-negativity.pdf', format='pdf', bbox_inches='tight')


"""
The next figures will scatter mutual entropy against fake entanglement.
This is done for the 4-qubit state with partitions 1000 and 1100
"""

for i in range(2):
	keys = ['1000', '1100']
	key = keys[i]
	plt.figure(figsize=fig_size)
	plt.scatter(ent_res['0011state'][key], neg_res['0011state'][key], s=scatter_size)
	plt.xlabel("Entropy")
	plt.ylabel("Negativity")
	#plt.title("Entropy vs negativity, {}".format(xlabels[i]))
	plt.show()
	#plt.savefig('Figures/Negativity results/entropy-vs-negativity-{}.pdf'.format(key), format='pdf', bbox_inches='tight')
	
	

print(average_negs)
print(std_negs)

print(average_ents)
print(std_ents)

print(average_fids)
print(std_fids)

plt.figure(figsize=fig_size)
plt.hist(neg_res['0011state']['1100'], bins=20)
plt.show()
#plt.savefig('Figures/Negativity results/negativity-histogram.pdf', format='pdf', bbox_inches='tight')

#2-3 qubits

average_negs_3 = np.average(neg_res['000_111_state']['100'])
average_conc = np.average(neg_res['00_11_state'])
average_negs = [average_negs_3, average_conc]

std_negs_3 = np.std(neg_res['000_111_state']['100'])
std_conc = np.std(neg_res['00_11_state'])
std_negs = [std_negs_3, std_conc]

sem_negs_3 = stats.sem(neg_res['000_111_state']['100'])
sem_conc = stats.sem(neg_res['00_11_state'])

average_ents_3 = np.average(ent_res['000_111_state']['100'])
average_ents_2 = np.average(ent_res['00_11_state'])
average_ents = [average_ents_3, average_ents_2]

std_ents_3 = np.std(ent_res['000_111_state']['100'])
std_ents_2 = np.std(ent_res['00_11_state'])
std_ents = [std_ents_3, std_ents_2]

sem_ents_3 = stats.sem(ent_res['000_111_state']['100'])
sem_ents_2 = stats.sem(ent_res['00_11_state'])

average_fids = [np.average(fid_res['000_111_state'])]
std_fids = [np.std(fid_res['000_111_state'])]
sem_fids = [stats.sem(fid_res['000_111_state'])]

data = [average_negs, average_ents]
xlabels = ['Negativity', 'Concurrence']


# create plot
fig, ax = plt.subplots(figsize=fig_size)
index = np.arange(2)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, data[0], bar_width,
yerr=std_negs,
alpha=opacity,
color='b',
label='Negativity/Concurrence')

rects2 = plt.bar(index + bar_width, data[1], bar_width,
yerr=std_ents,	
alpha=opacity,
color='g',
label='Mutual entropy')

#plt.xlabel('Partition')
plt.ylabel('Entropy/Negativity')
#plt.title('Negativity/Entropy of random states of given rank')
plt.xticks(index + bar_width/2, tuple(xlabels))
plt.legend(loc='center left')

plt.tight_layout()
plt.show()
#plt.savefig('Figures/Negativity results/2-and-3-qubit-negativity.pdf', format='pdf', bbox_inches='tight')