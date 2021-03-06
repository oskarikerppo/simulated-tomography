import numpy as np
#from qiskit.tools.qi.qi import partial_trace
from qutip import *
from scipy.optimize import minimize

class QDiscord():
    def __init__(self):
        self.paulikrons = self.get_pauli_kron_prods()

    def get_pauli_kron_prods(self):
        ## generates tensor products of Pauli matrices
        # define Pauli matrices:
        s0 = np.array([[1,0],[0,1]])
        sx = np.array([[0, 1],[ 1, 0]])
        sy = np.array([[0, -1j],[1j, 0]])
        sz = np.array([[1, 0],[0, -1]])
        paulis = np.array([s0,sx,sy,sz])

        # initialize tensor product list
        paulikrons = [[None for _ in range(4)] for __ in range(4)]
        for k1 in range(4):
            for k2 in range(4):
                paulikrons[k1][k2] = np.kron(paulis[k1],paulis[k2])
        
        return paulikrons

    def dm2cm(self, rho):
        # density matrix to correlation matrix
        corr_matrix = [[None for _ in range(4)] for __ in range(4)]
        for k1 in range(4):
            for k2 in range(4):
                corr_matrix[k1][k2] = np.real(np.trace(rho @ self.paulikrons[k1][k2])) # @ is matrix multiplication in numpy

        return np.array(corr_matrix)

    def discord(self, rho, minimize_discord = True, theta = np.pi/4., phi = np.pi/4., only_discord = True):
        """
        Calculates quantum discord and classical correlations on a 2-qubit state rho.
        Returns 
        """
        tau = np.array(self.dm2cm(rho))
        a = tau[1:4, 0]
        b = tau[0, 1:4].T 
        R = tau[1:4,1:4]
        
        n = lambda theta, phi: [np.sin(theta)*np.cos(phi),
                                np.sin(theta)*np.sin(phi),
                                np.cos(theta)]

        fn = lambda t, p: a @ n(t, p)
        RTn = lambda t, p: R.T @ n(t, p)
        gpn = lambda t, p: np.linalg.norm(b + RTn(t, p))
        gmn = lambda t, p: np.linalg.norm(b - RTn(t, p))
        
        pprob = lambda t, p: gpn(t,p)/(1. + fn(t,p))
        mprob = lambda t, p: gmn(t,p)/(1. - fn(t,p))


        ## Shannon entropy
        shan_ent = lambda x: - (1. + x)/2. * np.log2((1. + x)/2. + np.isclose(1. + x,0)) \
                             - (1. - x)/2. * np.log2((1. - x)/2. + np.isclose(1. - x,0))

        ## Conditional entropy 
        cond_ent = lambda t, p: (1. + fn(t,p)) / 2. * shan_ent(pprob(t, p)) +\
                                (1. - fn(t,p)) / 2. * shan_ent(mprob(t, p))

        ## VN entropy
        # Calculate eigenvalues
        eig = lambda rho: np.linalg.eigvals(rho) 
        vn_ent = lambda rho: - eig(rho).T @ np.log2(eig(rho) + np.isclose(eig(rho),0))
        #rhoA = partial_trace(rho, [1], dimensions=[2,2]) # get rho_A (trace out B)
        Qrho = Qobj(rho, dims=[[2, 2], [2, 2]])
        rhoA = np.array(Qrho.ptrace(0))
        #rhoB = partial_trace(rho, [0], dimensions=[2,2]) # get rho_B (trace out A)
        rhoB = np.array(Qrho.ptrace(1))

        ## Define discord and classical correlations
        if minimize_discord:
            cond_ent_temp = lambda tp: cond_ent(tp[0], tp[1])
            cond_ent_temp2 = minimize(cond_ent_temp, ([theta, phi])).fun
            qdiscord = vn_ent(rhoA) - vn_ent(rho) + cond_ent_temp2
            classical_corr = vn_ent(rhoB) - cond_ent_temp2
        else:
            tp = theta, phi
            qdiscord = vn_ent(rhoA) - vn_ent(rho) + cond_ent(tp[0], tp[1]) 
            classical_corr = vn_ent(rhoB) - cond_ent(tp[0], tp[1])             
        
        if only_discord:
            return qdiscord
        else:
            return qdiscord, classical_corr
