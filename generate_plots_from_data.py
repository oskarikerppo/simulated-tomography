import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import os
from scipy import stats
import plot_settings


#Initialize plot settings
PlotSettings = plot_settings.PlotSettings()

plt.rcParams.update(PlotSettings.tex_fonts)

fig_size = PlotSettings.set_size()



"""
Loads data files and produces plots.
Matplotlib settings are imported from plot_settings
"""

"""
Note: pathlib is used, because I tried to get the paths to work on both ubuntu and windows.
Currently I only use ubuntu so this is not guaranteed to work on macOS or windows.
Modify as needed until the data loads
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

max_1000 = 0

for key in neg_res['0011state'].keys():
	for x in neg_res['0011state'][key]:
		if x > max_1000:
			max_1000 = x

print(max_1000)

print("4-qubit maximum negativity:")
print("1000: {}".format(max(neg_res['0011state']['1000'])))
print("1100: {}".format(max(neg_res['0011state']['1100'])))


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
rects1 = plt.bar(index, data[0], PlotSettings.bar_width,
yerr=std_negs,
alpha=PlotSettings.opacity,
color='b',
label='Negativity')

#Rectangles for average mutual entropy with standard deviation
rects2 = plt.bar(index + PlotSettings.bar_width, data[1], PlotSettings.bar_width,
yerr=std_ents,	
alpha=PlotSettings.opacity,
color='g',
label='Mutual entropy')


plt.ylabel('Entropy/Negativity')
#plt.title('Negativity/Entropy of 4-qubit state')
plt.xticks(index + PlotSettings.bar_width/2, tuple(xlabels))
plt.legend()
plt.tight_layout()
plt.show()
#fig.savefig('Figures/Negativity results/4-qubit-negativity.pdf', format='pdf', bbox_inches='tight')


"""
The next figures will scatter mutual entropy against fake entanglement.
This is done for the 4-qubit state with partitions 1000 and 1100
"""

for i in range(2):
	keys = ['1000', '1100'] #Choose partitions
	key = keys[i]
	plt.figure(figsize=fig_size)
	plt.scatter(ent_res['0011state'][key], neg_res['0011state'][key], s=PlotSettings.scatter_size)
	plt.xlabel("Entropy")
	plt.ylabel("Negativity")
	#plt.title("Entropy vs negativity, {}".format(xlabels[i]))
	plt.show()
	#plt.savefig('Figures/Negativity results/entropy-vs-negativity-{}.pdf'.format(key), format='pdf', bbox_inches='tight')
	
	

#Next we produce the histogram of fake entanglement.


plt.figure(figsize=fig_size)
plt.hist(neg_res['0011state']['1100'], bins=PlotSettings.num_of_bins)
plt.show()
#plt.savefig('Figures/Negativity results/negativity-histogram.pdf', format='pdf', bbox_inches='tight')



"""
Next we produce the same plot as the first figure in this code, but for 2 and 3 qubits.
For two qubits we have fake concurrence and for 3 fake entanglement as usual.
"""

max_100 = 0
max_10 = 0

for key in neg_res['000_111_state'].keys():
	for x in neg_res['000_111_state'][key]:
		if x > max_100:
			max_100 = x


for x in neg_res['00_11_state']:
	if x > max_10:
		max_10 = x

print(max_100)
print(max_10)



print("2-3-qubit maximum negativity:")
print("100: {}".format(max(neg_res['000_111_state']['100'])))
print("10: {}".format(max(neg_res['00_11_state'])))


#Average negativity/concurrence
average_negs_3 = np.average(neg_res['000_111_state']['100'])
average_conc = np.average(neg_res['00_11_state'])
average_negs = [average_negs_3, average_conc]

#Standard deviation of fake negativities
std_negs_3 = np.std(neg_res['000_111_state']['100'])
std_conc = np.std(neg_res['00_11_state'])
std_negs = [std_negs_3, std_conc]

#Standard error of the mean of fake negativities
sem_negs_3 = stats.sem(neg_res['000_111_state']['100'])
sem_conc = stats.sem(neg_res['00_11_state'])
sem_negs = [sem_negs_3, sem_conc]

#Average mutual entropy
average_ents_3 = np.average(ent_res['000_111_state']['100'])
average_ents_2 = np.average(ent_res['00_11_state'])
average_ents = [average_ents_3, average_ents_2]

#Standard deviation of entropy
std_ents_3 = np.std(ent_res['000_111_state']['100'])
std_ents_2 = np.std(ent_res['00_11_state'])
std_ents = [std_ents_3, std_ents_2]

#Standard error of the mean of entropies
sem_ents_3 = stats.sem(ent_res['000_111_state']['100'])
sem_ents_2 = stats.sem(ent_res['00_11_state'])
sem_ents = [sem_ents_3, sem_ents_2]

#Average fidelities
average_fids = [np.average(fid_res['000_111_state'])]
std_fids = [np.std(fid_res['000_111_state'])]
sem_fids = [stats.sem(fid_res['000_111_state'])]

#Data for rectangles
data = [average_negs, average_ents]
xlabels = ['Negativity', 'Concurrence']


# create plot
fig, ax = plt.subplots(figsize=fig_size)
index = np.arange(2)

#Rectangle for negativity/concurrence
rects1 = plt.bar(index, data[0], PlotSettings.bar_width,
yerr=std_negs,
alpha=PlotSettings.opacity,
color='b',
label='Negativity/Concurrence')

#Rectangle for mutual entropy
rects2 = plt.bar(index + PlotSettings.bar_width, data[1], PlotSettings.bar_width,
yerr=std_ents,	
alpha=PlotSettings.opacity,
color='g',
label='Mutual entropy')


#plt.xlabel('Partition')
plt.ylabel('Entropy/Negativity')
#plt.title('Negativity/Entropy of random states of given rank')
plt.xticks(index + PlotSettings.bar_width/2, tuple(xlabels))
plt.legend(loc='center left')

plt.tight_layout()
plt.show()
#plt.savefig('Figures/Negativity results/2-and-3-qubit-negativity.pdf', format='pdf', bbox_inches='tight')








#Data for rectangles
data = [[average_conc, average_negs_3], [average_negs_31, average_negs_22]]
xlabels = ["1-to-1", "2-to-1", "3-to-1", "2-to-2"]


# create plot
fig, ax = plt.subplots(figsize=fig_size)
index = np.arange(2)

#Rectangle for negativity/concurrence
rects1 = plt.bar(index, data[0], PlotSettings.bar_width,
yerr=[std_conc, std_negs_3],
alpha=PlotSettings.opacity,
color=['g', 'b'],
label='Concurrence')

#Rectangle for mutual entropy
rects2 = plt.bar(index + PlotSettings.bar_width*1.4, data[1], PlotSettings.bar_width,
yerr=[std_negs_31, std_negs_22],	
alpha=PlotSettings.opacity,
color='b',
label='Negativity')


#plt.xlabel('Partition')
plt.ylabel('Concurrence/Negativity')
#plt.title('Negativity/Entropy of random states of given rank')
plt.xticks([0, 0.5, 1, 1.5], tuple(xlabels))
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
#plt.savefig('Figures/Negativity results/2-3-and-4-qubit-negativity.pdf', format='pdf', bbox_inches='tight')



#Data for rectangles
data = [[average_ents_2, average_ents_3], [average_ents_31, average_ents_22]]
xlabels = ["1-to-1", "2-to-1", "3-to-1", "2-to-2"]


# create plot
fig, ax = plt.subplots(figsize=fig_size)
index = np.arange(2)

#Rectangle for negativity/concurrence
rects1 = plt.bar(index, data[0], PlotSettings.bar_width,
yerr=[std_ents_2, std_ents_3],
alpha=PlotSettings.opacity,
color='b')
#label='Mutual entropy')

#Rectangle for mutual entropy
rects2 = plt.bar(index + PlotSettings.bar_width*1.4, data[1], PlotSettings.bar_width,
yerr=[std_ents_31, std_ents_22],	
alpha=PlotSettings.opacity,
color='b',
label='Mutual entropy')


#plt.xlabel('Partition')
plt.ylabel('Mutual entropy')
#plt.title('Negativity/Entropy of random states of given rank')
plt.xticks([0, 0.5, 1, 1.5], tuple(xlabels))
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
#plt.savefig('Figures/Negativity results/2-3-and-4-qubit-entropies.pdf', format='pdf', bbox_inches='tight')