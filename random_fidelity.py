from qutip import *
import numpy as np
import scipy
from scipy import stats
import itertools
import random
import matplotlib.pyplot as plt
import pickle
from time import time

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


width = 360
#'''
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)
#'''
#plt.rc('text', usetex = True)
fig_size = set_size(width)




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




A = 16
rank_lmt = 16
sample_size = 10000

init_state = smolin
#init_state = random_state(A, rank_lmt)
#init_state = random_qubit()
#init_state = rho([0,0,-1])

for A in range(1, 17):

	fids = []
	fids2 = []
	for i in range(sample_size):
		rand_state = rand_dm_ginibre(N=16, rank=A)
		#rand_state = random_state(A, rank_lmt)
		#rand_state = random_qubit()
		#fids.append(general_difelity(init_state, rand_state))
		fids2.append(fidelity(init_state, Qobj(rand_state)))

	#fids = np.array([fidelity(init_state, random_state(A, rank_lmt)) for x in range(sample_size)])

	#print(np.average(fids))
	#print(np.std(fids))
	avg = np.average(fids2)
	print(avg)
	std = np.std(fids2)
	print(std)

	#plt.hist(fids, 50)
	#plt.show()

	plt.figure(figsize=fig_size)
	plt.hist(fids2, 50)
	plt.savefig('Figures/Random fidelity/random-fidelity-histogram-rank-{}-avg-{}-std-{}.pdf'.format(A, avg, std), format='pdf', bbox_inches='tight')
	plt.show()

	#print(init_state)
	#print(general_difelity(init_state, init_state))



