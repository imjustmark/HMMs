#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import numpy as np
from math import log

def transpose(B):
    num_cols_B = len(B[0])
    B_t = []
    for _ in range(0,num_cols_B):
        B_t.append([])
    for row in B:
        for idx, elem in enumerate(row):
            B_t[idx].append(elem)
    return B_t

def matrix_multiplication(A, B):
    # if A is a vector, it is a row vector.
    # if B is a vector, it is a column vector.
    vectorB = False
    if type(A[0]) != list:
        A = [A]
    if type(B[0]) != list:
        vectorB = True
        new_B = []
        for elem in B:
         new_B.append([elem])
        B = new_B
    num_rows_A = len(A)
    num_cols_A = len(A[0])
    num_rows_B = len(B)
    if num_rows_B != num_cols_A:
        print("NOT MULTIPLIABLE!!!!")
        return
    else:
        res = []
        if not vectorB:
            for _ in range(0,num_rows_A):
                res.append([])
        B_t = transpose(B)
        for i, rowA in enumerate(A):
            for j, rowB in enumerate(B_t):
                res_sum = 0
                for idx in range(0,num_cols_A):
                    valA = rowA[idx]
                    valB = rowB[idx]
                    res_sum += valA*valB
                if vectorB:
                    res.append(res_sum)
                else:
                    res[i].append(res_sum)
        if not(vectorB) and len(res) == 1:
            res = res[0]
        return res

def hadamard_product(A,B):
    result = []
    for i in range(0,len(A)):
        result.append(A[i]*B[i])
    return result

def normalize(vec):
    c = sum(vec)
    res = []
    for elem in vec:
        res.append(elem/c)
    return res, 1/c

def floor_observation_pdf(column):
    floor = 10e-100
    change = True
    for elem in column:
        if elem > floor:
            change = False
    if change:
        factor = [floor] * len(column)
    else:
        factor = column
    return factor

def forward_algorithm(A,B,pi,obs):
    if len(obs) <= 0:
        return 0
    B_t = transpose(B)
    factor = floor_observation_pdf(B_t[obs[0]])
    alpha = hadamard_product(pi[0], factor)
    if all(x == 0 for x in alpha):
        return None, None, False
    alpha, c = normalize(alpha)
    new_cs = [c]
    alphas = [alpha]
    A_t = transpose(A)
    for o in obs[1:]:
        factor = floor_observation_pdf(B_t[o])
        # eprint(factor)
        alpha = hadamard_product(matrix_multiplication(alpha, A), factor)
        # eprint(alpha, "\n")
        if all(x == 0 for x in alpha):
            return None, None, False
        alpha, c = normalize(alpha)
        alphas.append(alpha)
        new_cs.append(c)
    return alphas, new_cs, True

def backward_algorithm(A,B,pi,obs,cs):
    if len(obs) <= 0:
        return 0
    B_t = transpose(B)
    beta = [cs[-1] for _ in pi[0]]
    betas = [beta]
    new_obs = obs.copy()
    new_obs.reverse()
    t = len(cs)-2
    for o in new_obs[:-1]:
        beta = matrix_multiplication(hadamard_product(beta, B_t[o]), transpose(A))
        beta = [x * cs[t] for x in beta]
        betas.append(beta)
        t -= 1
    betas.reverse()
    return betas

def compute_di_gamma(alphas, betas, A, B, obs):
    di_gammas = []
    # norm = sum(alphas[-1])
    for t in range(len(obs)-1):
        gamma_t = []
        for i in range(len(alphas[0])):
            row = []
            alpha_t_i = alphas[t][i]
            for j in range(len(betas[0])):
                beta_t1 = betas[t+1][j]
                a_i_j = A[i][j]
                b_j_t1 = B[j][obs[t+1]]
                result = alpha_t_i * a_i_j * beta_t1 * b_j_t1
                row.append(result)
            gamma_t.append(row)
        di_gammas.append(gamma_t)
    return di_gammas

def compute_gamma(di_gammas, alphas):
    gammas = []
    last_alpha = alphas[-1]
    for t in range(len(di_gammas)):
        row = []
        for i in range(len(di_gammas[t])):
            gamma_i_t = sum(di_gammas[t][i])
            row.append(gamma_i_t)
        gammas.append(row)
    gammas.append(last_alpha)
    return gammas


def estimate_parameters(di_gammas, gammas, obs, refB):
    A = []
    B = []
    pi = []
    gammas_T = transpose(gammas)
    norm_factors = []
    for i in range(len(gammas[0])):
        row = []
        norm = sum(gammas_T[i][:-1])
        for j in range(len(gammas[0])):
            res_a = 0
            for t in range(len(gammas)-1):
                res_a += di_gammas[t][i][j] / norm
            row.append(res_a)
        norm_factors.append(norm + gammas_T[i][-1])
        A.append(row)

        sums = [0] * len(refB[0])
        for t in range(len(gammas)):
            o = obs[t]
            sums[o] += gammas[t][i] / norm_factors[i]
        B.append(sums)

        pi.append(gammas[0][i])
    return A, B, [pi]

def compute_probability(cs):
    log_prob = 0
    for c in cs:
        log_prob += log(c)
    log_prob = -log_prob
    return log_prob

def baum_welch(max_iters, A, B, pi, obs):
    iters = 0
    # probs = [-1000000000001, -1000000000000]
    # while probs[iters] < probs[iters+1] and iters < max_iters:
    while iters < max_iters:
        iters += 1
        alphas, cs, _ = forward_algorithm(A,B,pi,obs)
        betas = backward_algorithm(A,B,pi,obs,cs)
        di_gammas = compute_di_gamma(alphas, betas, A, B, obs)
        gammas = compute_gamma(di_gammas, alphas)
        A, B, pi = estimate_parameters(di_gammas, gammas, obs, B)
        # current_prob = compute_probability(cs)
        # probs.append(current_prob)
    return A, B, pi, cs, iters


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        # self.A = (np.random.randn(N_SPECIES, N_SPECIES)*0.4 + 0.5).tolist()
        # self.B = (np.random.randn(N_SPECIES, N_EMISSIONS)*0.4 + 0.5).tolist()
        # self.pi = (np.random.randn(1, N_SPECIES)*0.4 + 0.5).tolist()
        """
        self.A = np.random.beta(2, 2, (N_SPECIES, N_SPECIES))
        self.B = np.random.beta(2, 2, (N_SPECIES, N_EMISSIONS))
        self.pi = np.random.beta(2, 2, (1, N_SPECIES))
        sum_A = np.sum(self.A, axis=1)
        self.A = (self.A / sum_A[:,np.newaxis]).tolist()
        sum_B = np.sum(self.B, axis=1)
        self.B = (self.B / sum_B[:,np.newaxis]).tolist()
        sum_pi = np.sum(self.pi, axis=1)
        self.pi = (self.pi / sum_pi[:,np.newaxis]).tolist()

        self.max_iters = 30
        self.train_steps = 100
        self.current_fish = 0
        self.cs = None
        """
        self.models = []    # Store the parameters of the models
        self.guesses = []   # Store the species to guess using each model
        self.correct_unused_indices = list(range(N_SPECIES))  # Store which species we haven't guessed yet
        self.hmms_used_to_guess = []    # Stores the hmms used to guess each fish

        # HYPERPARAMETERS
        self.k = 3              # Threshold quality of predictions
        self.max_iters = 10     # Number of iterations of model training (Baum-Welch)
        self.num_states = 5     # Number of states of species' HMMs.

        self.observation_steps = N_STEPS - 70
        self.current_fish = -1
        self.observations = []
        self.observationsT = []
        # self.train_probabilities = []
        # self.probabilities = []

        # self.num_guesses = 0

    def add_new_hmm(self):
        A = np.random.beta(2, 2, (self.num_states, self.num_states))
        B = np.random.beta(2, 2, (self.num_states, N_EMISSIONS))
        pi = np.random.beta(2, 2, (1, self.num_states))
        sum_A = np.sum(A, axis=1)
        A = (A / sum_A[:,np.newaxis]).tolist()
        sum_B = np.sum(B, axis=1)
        B = (B / sum_B[:,np.newaxis]).tolist()
        sum_pi = np.sum(pi, axis=1)
        pi = (pi / sum_pi[:,np.newaxis]).tolist()

        A, B, pi, train_cs, iters = baum_welch(self.max_iters, A, B, pi, self.observationsT[self.current_fish])
        train_prob = compute_probability(train_cs)
        # We keep track of the probability of generating the list of observations used to train the hmm
        # as a reference to be used in the ref_threshold function.
        self.models.append((A,B,pi,train_prob))
        # If there are no unused species left to guess, we guess the one that yielded highest probability.
        if len(self.correct_unused_indices) == 0:
            self.guesses.append(self.guesses[self.idx_max])
        else:
            # If there are, we select one unused species at random and we guess it.
            self.guesses.append(self.correct_unused_indices.pop())
        
    def ref_threshold(self, scores, k):
        candidates = []
        for idx, hmm in enumerate(self.models):
            # eprint("Reference: " + str(hmm[3]))
            # eprint("Probability: " + str(scores[idx]))
            # If the probability of the current observations to have been generated by an hmm
            # is greater than k times the probability of the observation sequence used to train
            # that hmm, then we consider that hmm as a candidate.
            if scores[idx] > k * hmm[3]:
                candidates.append(scores[idx])
        if len(candidates) > 0:
            # And we select the one with highest probability among the candidates.
            return scores.index(max(candidates))
        else:
            return None

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        """
        # States are Species Implementation
        if step < self.train_steps:
            eprint(observations)
            self.A, self.B, self.pi, self.cs = baum_welch(self.max_iters, self.A, self.B, self.pi, observations, self.cs)
            return None
        else:
            sequence = viterbi_algorithm(self.A,self.B,self.pi,observations)
            self.current_fish +=1
            return (self.current_fish-1, sequence[self.current_fish-1])
        """
        # We collect fish observations during the first observation_steps steps.
        if step < self.observation_steps:
            self.observations.append(observations)
        else:
            # During the last T - observations_steps steps, we go fish by fish
            # using their collected observations.
            self.current_fish += 1
            if step == self.observation_steps:
                self.observationsT = transpose(self.observations)
            current_obs = self.observationsT[self.current_fish]
            max_prob = -1000000
            self.idx_max = -1
            probs = []
            # For each hmm we have to this point, we compute the probability that the observations
            # of the current fish were generated by that hmm. We keep track of which hmm yielded the
            # highest probability.
            for idx, hmm in enumerate(self.models):
                _, cs, correct = forward_algorithm(hmm[0], hmm[1], hmm[2], current_obs)
                if correct:
                    probability = compute_probability(cs)
                else:
                    probability = -np.inf
                probs.append(probability)
                if probability > max_prob:
                    max_prob = probability
                    self.idx_max = idx
            # If there is an hmm that satisfies our condition to be a candidate...
            candidate_idx = self.ref_threshold(probs, self.k)

            # We guess the species associated with that hmm.
            if candidate_idx is not None:
                # eprint('Suited! ' + str(self.guesses[candidate_idx]))
                self.hmms_used_to_guess.append((candidate_idx, True))
                # self.num_guesses += 1
                return (self.current_fish, self.guesses[candidate_idx])
            # If not, we train a new hmm for that fish.
            else:
                self.add_new_hmm()
                self.hmms_used_to_guess.append((len(self.models)-1, False))
                # eprint('New Model! ' + str(self.guesses[-1]))
                # self.num_guesses += 1
                return (self.current_fish, self.guesses[-1])
                
    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.s
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        # The boolean "suited" tells us if we trained a new HMM for the current fish (False)
        # or if we used a previously trained one (True).
        hmm_used, suited = self.hmms_used_to_guess[-1]
        # eprint(true_type)
        if suited and not correct:
            # If we suited and missed we train a new hmm for the fish and keep its true species.
            self.add_new_hmm()
            # eprint("Oh, we were wrong...")
            self.guesses[-1] = true_type
        elif not suited:
            # If we had trained a new hmm, we update its guess with the true species.
            self.guesses[hmm_used] = true_type
        