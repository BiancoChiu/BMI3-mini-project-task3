import numpy as np

from utils import *


class HMM:
    def __init__(self, N, M, transition=None, emission=None, initial=None):
        if transition is None:
            transition = np.ones((N, N))
            self.log_transition = np.log(transition / np.sum(transition, axis=1, keepdims=True))
        else:
            self.log_transition = np.log(transition / np.sum(transition, axis=1, keepdims=True))
            
        if emission is None:
            emission = np.ones((N, M))
            self.log_emission = np.log(emission / np.sum(emission, axis=1, keepdims=True))
        else:
            self.log_emission = np.log(emission / np.sum(emission, axis=1, keepdims=True))

        if initial is None:
            initial = np.ones(N)
            self.log_initial = np.log(initial / np.sum(initial))
        else:
            self.log_initial = np.log(initial / np.sum(initial, axis=1, keepdims=True))
    
    def forward_log(self, obs):
        T = obs.shape[1]
        N = self.log_transition.shape[0]
        
        alpha = np.full((N, T), -np.inf)
        alpha[:, 0] = self.log_initial.flatten() + self.log_emission[:, obs[0, 0]]
        
        for t in range(1, T):
            for j in range(N):
                alpha[j, t] = np.logaddexp.reduce(
                    alpha[:, t-1] + self.log_transition[:, j]
                ) + self.log_emission[j, obs[0, t]]

        log_prob = np.logaddexp.reduce(alpha[:, -1])
        return log_prob, alpha
    
    def backward_log(self, obs):
        T = obs.shape[1]
        N = self.log_transition.shape[0]

        beta = np.full((N, T), -np.inf)
        beta[:, -1] = 0.0
        
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[i, t] = np.logaddexp.reduce(
                    beta[:, t+1] + 
                    self.log_transition[i, :] + 
                    self.log_emission[:, obs[0, t+1]]
                )
        
        return beta
    
    def viterbi_log(self, obs):
        T = obs.shape[1]
        N = self.log_transition.shape[0]
        
        Delta = np.full((N, T), -np.inf)
        Psi = np.zeros((N, T), dtype=int)

        Delta[:, 0] = self.log_initial.flatten() + self.log_emission[:, obs[0, 0]]
        
        for t in range(1, T):
            for j in range(N):
                temp = Delta[:, t-1] + self.log_transition[:, j]
                Delta[j, t] = np.max(temp) + self.log_emission[j, obs[0, t]]
                Psi[j, t] = np.argmax(temp)
        
        P_best = np.max(Delta[:, -1])
        
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(Delta[:, -1])
        
        for t in range(T-2, -1, -1):
            path[t] = Psi[path[t+1], t+1]
        
        return path, np.exp(P_best)
    
    def baum_welch_log(self, obs, n_iter):
        N = self.log_transition.shape[0]
        M = self.log_emission.shape[1]
        T = obs.shape[1]
        epsilon = 1e-10
        
        for _ in range(n_iter):
            log_prob, alpha = self.forward_log(obs)
            
            beta = self.backward_log(obs)
            log_posterior = alpha + beta - log_prob
            Gamma = np.exp(log_posterior)
            
            Xi = np.zeros((T-1, N, N))
            for t in range(T-1):
                for i in range(N):
                    for j in range(N):
                        Xi[t, i, j] = np.exp(
                            alpha[i, t] + 
                            self.log_transition[i, j] + 
                            self.log_emission[j, obs[0, t+1]] + 
                            beta[j, t+1] - 
                            log_prob
                        )
            
            new_initial = Gamma[:, 0] / np.sum(Gamma[:, 0])
            self.log_initial = np.log(new_initial + epsilon)
            
            for i in range(N):
                gamma_sum = np.sum(Gamma[i, :-1])
                for j in range(N):
                    xi_sum = np.sum(Xi[:, i, j])
                    new_prob = (xi_sum + epsilon) / (gamma_sum + epsilon)
                    self.log_transition[i, j] = np.log(new_prob)
            
            for j in range(N):
                for k in range(M):
                    numer = np.sum(Gamma[j, np.where(obs[0, :] == k)[0]])
                    denom = np.sum(Gamma[j, :])
                    
                    new_prob = (numer + epsilon) / (denom + epsilon)
                    self.log_emission[j, k] = np.log(new_prob)
        
        return np.exp(self.log_transition), np.exp(self.log_emission), np.exp(self.log_initial)

    def adjust_emission_matrix(self, selected_proteins):
        num_proteins = 4
        old_obs_space = range(2 ** num_proteins)  # 0, 1, ..., 15

        selected_indices = [i for i, bit in enumerate(selected_proteins[::-1]) if bit == '1']
        print(f'selected_indices: {selected_indices}')
        num_selected = len(selected_indices)
        
        new_obs_space = range(2 ** num_selected)
        old_to_new_map = {}

        for old_obs in old_obs_space:
            new_obs = 0
            for i, selected_idx in enumerate(selected_indices):
                if old_obs & (1 << selected_idx):
                    new_obs |= (1 << i)
            old_to_new_map[old_obs] = new_obs

        N, M_old = self.log_emission.shape
        M_new = 2 ** num_selected
        log_emission_new = np.full((N, M_new), -np.inf)

        for old_obs, new_obs in old_to_new_map.items():
            log_emission_new[:, new_obs] = np.logaddexp(
                log_emission_new[:, new_obs],
                self.log_emission[:, old_obs]
            )

        self.log_emission = log_emission_new
        print("emission matrix has been adjusted according to the selected proteins")

    # def predict_hidden_states(self, observations, selected_proteins):
        # obs_map=generate_obs_map(selected_proteins)
        # obs_sequence = np.array([obs_map[o] for _,o in np.ndenumerate(observations)])
        # path, path_prob = self.viterbi_log(observations.reshape(1, -1))
        # return path, path_prob

if __name__ == '__main__':
    np.random.seed(42)

    transition = np.array([[0.6, 0.4], [0.6, 0.4]])
    emission = np.array([[1 / 16] * 16, [1 / 16] * 16])
    initial = np.array([[0.5, 0.5]])
    log_transition = np.log(transition)
    log_emission = np.log(emission)
    log_initial = np.log(initial)
    hmm = HMM(2, 16, log_transition, log_emission, log_initial)
    hmm.adjust_emission_matrix('1000')
    print(hmm.log_emission.shape)
