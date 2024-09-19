#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
from custom_functions import *
import numpy as np



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

        self.observation_steps = 100
        self.current_fish = -1
        self.observations = []
        self.observationsT = []
        self.train_probabilities = []
        self.probabilities = []

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
        self.models.append((A,B,pi,train_prob))
        if len(self.correct_unused_indices) == 0:
            self.guesses.append(self.guesses[self.idx_max])
        else:
            self.guesses.append(self.correct_unused_indices.pop())
        
    def ref_threshold(self, scores, k):
        candidates = []
        for idx, hmm in enumerate(self.models):
            # eprint("Reference: " + str(hmm[3]))
            # eprint("Probability: " + str(scores[idx]))
            if scores[idx] > k * hmm[3]:
                candidates.append(scores[idx])
        if len(candidates) > 0:
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
        if step < self.observation_steps:
            self.observations.append(observations)
        else:
            self.current_fish += 1
            eprint("Step: " + str(step) + " Guesses: " + str(self.num_guesses))
            if step == self.observation_steps:
                self.observationsT = transpose(self.observations)
            current_obs = self.observationsT[self.current_fish]
            max_prob = -1000000
            self.idx_max = -1
            probs = []
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
            candidate_idx = self.ref_threshold(probs, self.k)

            if candidate_idx is not None:
                # eprint('Suited! ' + str(self.guesses[candidate_idx]))
                self.hmms_used_to_guess.append((candidate_idx, True))
                self.num_guesses += 1
                return (self.current_fish, self.guesses[candidate_idx])
            else:
                self.add_new_hmm()
                self.hmms_used_to_guess.append((len(self.models)-1, False))
                # eprint('New Model! ' + str(self.guesses[-1]))
                self.num_guesses += 1
                return (self.current_fish, self.guesses[-1])
                
    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        hmm_used, suited = self.hmms_used_to_guess[-1]
        # eprint(true_type)
        if suited and not correct:
            self.add_new_hmm()
            # eprint("Oh, we were wrong...")
            self.guesses[-1] = true_type
        elif not suited:
            self.guesses[hmm_used] = true_type
        
