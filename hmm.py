from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])

        alpha[:,0] = self.pi*self.B[:,self.obs_dict[Osequence[0]]]

        for i in range(1, L):
            for s in range(S):
                alpha[s,i] = self.B[s, self.obs_dict[Osequence[i]]]*(np.sum(self.A[:,s]*alpha[:,i-1]))

        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])

        beta[:,L-1] = 1

        for i in reversed(range(L-1)):
            for s in range(S):
                beta[s,i] = np.sum(self.A[s].T*self.B[:,self.obs_dict[Osequence[i+1]]]*beta[:,i+1])

        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        prob = np.sum(self.forward(Osequence)[:,len(Osequence)-1])
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        denominator = self.sequence_prob(Osequence)

        prob = alpha*beta/denominator
        return prob

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        denominator = self.sequence_prob(Osequence)

        for i in range(L-1):
            for s_ in range(S):
                prob[:,s_,i] = alpha[:, i]*self.A[:,s_]*self.B[s_, self.obs_dict[Osequence[i+1]]]*beta[s_, i+1]/denominator

        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        path_index = np.zeros([len(Osequence)], dtype='int')

        S = len(self.pi)
        L = len(Osequence)
        delta_prob = np.zeros([S, L])
        delta = np.zeros((S, L), dtype='int')

        delta_prob[:,0] = self.pi*self.B[:,self.obs_dict[Osequence[0]]]

        for i in range(1,L):
            for s in range(S):
                delta_prob[s, i] = self.B[s,self.obs_dict[Osequence[i]]]*np.max(self.A[:,s]*delta_prob[:,i-1])
                delta[s,i] = np.argmax(self.A[:,s]*delta_prob[:,i-1])

        path_index[L-1] = np.argmax(delta_prob[:,L-1])

        for i in reversed(range(1, L)):
            path_index[i-1] = delta[path_index[i], i]

        state_keys = list(self.state_dict.keys())
        state_index = list(self.state_dict.values())

        path = [state_keys[state_index.index(i)] for i in path_index]

        return path



