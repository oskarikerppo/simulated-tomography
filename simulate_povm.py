from qutip import *
import numpy as np
import scipy
import itertools
import random
import matplotlib.pyplot as plt
import pickle
from time import time


#Pauli matrices
#s = [sigmax(), sigmay(), sigmaz()]
s = {0: sigmax(), 1: sigmay(), 2: sigmaz()}

#General qubit state, input as list of Bloch vector components, i.e. r = [rx, ry, rz]
def rho(r):
	if np.linalg.norm(r) != 1:
		r = np.array(r)/np.linalg.norm(r)
	return (qeye(2) + sum([r[i] * s[i] for i in range(3)])) / 2

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
			#d[i][j] = elem
			d[j][i] = elem
	#set complex elements
	for i in range(3):
		for j in range(i+1, 4):
			elem = p.pop(0)
			#d[i][j] += elem*(-1j)
			d[j][i] += elem*(1j)
	d = d.T.conj() @ d
	return d

def density_to_vector(density):#p is list of 16 numbers
	d = np.array(density, dtype=complex)
	p = []
	#Set diagonal elements
	for i in range(4):
		#d[i][i] = p.pop(0)
		p.append(np.real(d[i][i]))
	#set real elements
	for i in range(3):
		for j in range(i+1, 4):
			p.append(np.real(d[i][j]))
	#set complex elements
	for i in range(3):
		for j in range(i+1, 4):
			p.append(np.imag(d[i][j]))
	return tuple(p)

#Convert simulated statistics to 2-qubit statistics
def convert_statistics(statistics, q1, q2, M, M2):
	conv_stat = []
	for i in range(len(statistics)):
		s = M[statistics[i]]
		c_s = s[q1] + s[q2]
		conv_stat.append(M2.index(c_s))
	return conv_stat


def maximum_likelihood(p, m_2_obs, frequencies):
	density_matrix = qubit2density(p)
	max_sum = 0
	for i in range(len(m_2_obs)):
		s = np.real(np.trace(m_2_obs[i] * density_matrix))
		if s != 0:
			max_sum += frequencies.count(i)*np.log(s)
	return np.real(-max_sum/len(frequencies))

#Function for fitting
def func(x, a, b, c):
	return a - b / np.log(c * x)
 
# Log-likelihood function
def log_l(rho, observables, counts):
    return sum([counts[obs] * np.log(np.real((observables[obs] @ rho).trace())) for obs in counts]) / sum(counts.values())

# R(rho) operator
def R(rho, observables, counts):
    R = np.zeros((rho.shape[0], rho.shape[0]), dtype=complex)
    for obs in counts:
        R += (counts[obs] / (observables[obs] @ rho).trace()) * observables[obs]
    R /= sum(counts.values())
    return R

# Returns rho^{(k+1)} given rho (not diluted)
def RrR(rho, observables, marginals):
    rhok = R(rho, observables, marginals) @ rho @ R(rho, observables, marginals)
    return rhok / rhok.trace()

# Returns rho^{(k+1)} given rho and epsilon
def IRrR(rho, observables, marginals, epsilon):
    M = (np.eye(len(rho)) + epsilon * R(rho, observables, marginals)) / (1 + epsilon)
    rhok = M @ rho @ M
    return rhok / rhok.trace()

def marginalize(marginals, qlist):
	new_marginals = {}
	for key in marginals.keys():
		new_key = ''
		for i in range(len(qlist)):
			new_key += key[qlist[i]]
		outcomes = new_marginals.get(new_key, 0)
		outcomes += marginals[key]
		new_marginals[new_key] = outcomes
	return new_marginals

# Maximises the log-likelihood
def infer_state(marginals, qlist, observables, tol=1e-15, maxiter=1000, epsilon_range=1e-8, n_epsilons=1000):
    """
    Returns the state that maximises the log-likelihood given the observations.
    input:
        marginals (dict): dictionary with the marginal counts for all the groups of k qubits under consideration.
        qlist (tuple): qubits for which the maximisation is carried out.
        observables (dict): dictionary with the effect (numpy array) corresponding to each outcome.
        tol (float): tolerance for the convergence of the algorithm.
        maxiter (int): maximum number of iterations.
        epsilon_range (float): range in which random values of epsilon are sampled in the second and third phases.
        n_epsilons (int): number of values of epsilon in the range (0, epsilon_range] to be maximised over in phase 3.
        
        The format for the keys in 'marginals' and 'observables' is a chain of outcomes for the given POVM with 
        the same order as qlist. For instance, '031' corresponds to qlist[0] with outcome '1', qlist[1] yielding '3',
        and qlist[2], '0'.
    output:
        A density matrix (numpy array).
    """
    
    # Number of qubits
    k = len(qlist)
    
    marginals = marginalize(marginals, qlist)


    # Phase 1: iterative algorithm without (not diluted)
    rhok = np.eye(2**k) / 2**k
    for iteration_one in range(maxiter):
        rho = rhok
        rhok = RrR(rho, observables, marginals)
        if log_l(rhok, observables, marginals) < log_l(rho, observables, marginals):
            # Stop if likelihood decreases (and do not accept last step)
            rhok = rho
            break
        elif np.isclose(log_l(rhok, observables, marginals), log_l(rho, observables, marginals), atol=tol, rtol=0) and np.isclose(rhok, rho, atol=tol, rtol=0).all():
            # Stop if increase in likelihood and rhok-rho are small enough
            break

    # Phase 2: iterate diluted algorithm with random epsilon
    for iteration_two in range(maxiter):
        rho = rhok
        epsilon = np.random.rand() * epsilon_range
        rhok = IRrR(rho, observables, marginals, epsilon)
        if log_l(rhok, observables, marginals) < log_l(rho, observables, marginals):
            # If likelihood decreases, do not accept the change but continue
            rhok = rho
        elif np.isclose(log_l(rhok, observables, marginals), log_l(rho, observables, marginals), atol=tol, rtol=0) and np.isclose(rhok, rho, atol=tol, rtol=0).all():
            # Stop if increase in likelihood and rhok-rho are small enough
            break
    
    # Phase 3: iterate dilute algorithm for largest value of epsilon
    epsilons = np.linspace(0, epsilon_range, n_epsilons+1)[1:]
    for iteration_three in range(maxiter):
        # Find largest increase in log-likelihood
        delta_logl = {epsilon: log_l(IRrR(rhok, observables, marginals, epsilon), observables, marginals) - log_l(rhok, observables, marginals) for epsilon in epsilons}
        max_epsilon = max(delta_logl, key=delta_logl.get)
        if delta_logl[max_epsilon] > tol:
            rhok = IRrR(rhok, observables, marginals, epsilon)
        else:
            break
    
    # Verify result
    delta_logl = {epsilon: log_l(IRrR(rhok, observables, marginals, epsilon), observables, marginals) - log_l(rhok, observables, marginals) for epsilon in epsilons}
    if not (max(delta_logl.values()) < tol and np.isclose(log_l(rhok, observables, marginals), log_l(rho, observables, marginals), atol=tol, rtol=0) and np.isclose(rhok, rho, atol=tol, rtol=0).all()):
        print('Convergence not achieved:')
        print('Delta log-likelihood:', np.abs(log_l(rhok, observables, marginals) - log_l(rho, observables, marginals)))
        print('Largest difference in operators:', np.amax(np.abs(rho - rhok)))
        print('Iterations:')
        print('Phase 1:', iteration_one+1)
        print('Phase 2:', iteration_two+1)
        print('Phase 3:', iteration_three+1)
    
    return rhok


def calculate_expectation(POVM, input_state, outcomes, i, n):
	effects = [POVM[int(outcomes[i][j])] for j in range(n)]
	return np.real((tensor(effects) * input_state).tr())


def main(n, k, qlist, input_state, POVM, expectations, state_name, meas_name, seed=False):
	
	#s0 = time()

	if seed:
		random.seed(seed)

	povm_string = ""
	for i in range(len(POVM)):
		povm_string += str(i)

	#print("Listing outcomes")
	#N-qubit POVM from the SIC-POVMs
	M = {}
	outcomes = []
	for i, item in enumerate(itertools.product(povm_string, repeat=n)):
		M["".join(item)] = 0
		outcomes.append("".join(item))

	#s1 = time()
	#print(s1 - s0)
	#s2 = time()
	
	#print("Simulating statistics")
	#Simulate outcome statistics
	
	sim = np.random.choice(outcomes, size=k, p=expectations)
	#s25 = time()
	#print(s25 - s2)
	#print("Assigning values to dict")
	for i in sim:
		outcomes = M.get(i, 0)
		outcomes += 1
		M[i] = outcomes

	#s3 = time()
	#print(s3 - s25)
	#Reconstruct k-qubit states from outcome statistic
	k_wise = len(qlist)
	

	#print("Listing k-qubit outcomes and forming effects")
	#k-qubit optimization, maximum likelihood

	#k-qubit observable
	M2 = {}
	outcomes2 = []
	for i, item in enumerate(itertools.product(povm_string, repeat=k_wise)):
		M2["".join(item)] = 0
		outcomes2.append("".join(item))

	m_2_obs = {}
	for i in range(len(outcomes2)):
		effects = [POVM[int(outcomes2[i][j])] for j in range(k_wise)]
		m_2_obs[outcomes2[i]] = np.array(tensor(effects))


	#s4 = time()
	#print(s4 - s3)
	if k_wise < n:
 		partial_W = Qobj(np.array(input_state.ptrace(list(qlist))))
	else:
		partial_W = input_state
	#print("Inferring state")
	sol_den =	infer_state(M, qlist, m_2_obs, tol=1e-15, maxiter=1000, epsilon_range=1e-8, n_epsilons=1000)
	#s5 = time()
	#print(s5 - s4)
	sol_den = Qobj(sol_den)
	fid = fidelity(partial_W, sol_den)
	#print(fid)
	#print("Total time: {} minutes".format((time() - s0)/60))
	return fid, sol_den, partial_W

def w_state(n):
	w_vec = []
	for i in range(n):
		vec = []
		for j in range(n):
			if i == j:
				vec.append(basis(2, 0))
			else:
				vec.append(basis(2, 1))
		##print(tensor(vec))
		w_vec.append(tensor(vec))		
	w_vec = sum(w_vec).unit()

	#Density matrix for W state
	w_state = w_vec * w_vec.dag()
	return w_state

if __name__ == "__main__":
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

	init_state = w_state(8)

	#s0= time()
	#print("Initializing")
	povm_string = ""
	for i in range(len(E)):
		povm_string += str(i)

	outcomes = []
	for i, item in enumerate(itertools.product(povm_string, repeat=8)):
		outcomes.append("".join(item))

	expectations = np.array([calculate_expectation(E, init_state, outcomes, i, 8) for i in range(len(outcomes))])
	#s1 = time()
	#print(s1 - s0)

	fids2 = []
	fids3 = []
	fids4 = []
	for i in range(10):
		fids4.append(main(8, 8192, (1, 4, 5, 7), init_state, E, expectations, "W", "sic", seed=False))
		fids3.append(main(8, 8192, (1, 4, 7), init_state, E, expectations, "W", "sic", seed=False))
		fids2.append(main(8, 8192, (4, 7), init_state, E, expectations, "W", "sic", seed=False))
	#print(np.std(fids2))
	#print(np.average(fids2))
	#print(np.std(fids3))
	#print(np.average(fids3))
	#print(np.std(fids4))
	#print(np.average(fids4))


