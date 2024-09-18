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
        self.models = []
        self.observation_steps = 110
        self.max_iters = 100
        self.threshold = 0.9
        self.observations = []
        self.observationsT = []
        self.num_states = 5
        self.current_fish = -1
        self.percentage = 0.7
        self.train_probabilities = []
        self.k = 5
        self.guesses = []
        self.used_hmm = {}

        self.probabilities = []

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
        self.models.append([A, B, pi])
        self.guesses.append(len(self.models)-1)

    def avg_threshold(self, scores, percentage):
        if len(scores) == 1:
            return None
        avg = np.average([x for x in scores if x != -np.inf])
        candidates = []
        threshold = percentage * avg
        for elem in scores:
            if elem > avg - threshold:
                candidates.append(elem)
        if len(candidates) > 0:
            return scores.index(max(candidates))
        else:
            return None
        
    def ref_threshold(self, scores, k):
        candidates = []
        for idx, elem in enumerate(scores):
            if elem > k * self.train_probabilities[idx]:
                candidates.append(elem)
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
            if step == self.observation_steps:
                self.observationsT = transpose(self.observations)
            current_obs = self.observationsT[self.current_fish]
            max_prob = -1000000
            idx_max = -1
            probs = []
            for idx, hmm in enumerate(self.models):
                _, cs, correct = forward_algorithm(hmm[0], hmm[1], hmm[2], current_obs)
                if correct:
                    probability = compute_probability(cs)
                else:
                    probability = -np.inf
                #probability = sum(alphas[-1])/cs[-1]
                probs.append(probability)
                if probability > max_prob:
                    max_prob = probability
                    idx_max = idx
            idx_candidate = self.ref_threshold(probs, self.k)
            eprint(probs)
            if idx_candidate is not None:
                eprint('Suited! ' + str(idx_candidate))
                self.used_hmm[self.current_fish] = idx_candidate
                return (self.current_fish, idx_candidate)
            elif len(self.models) < N_SPECIES:
                self.add_new_hmm()
                self.models[-1][0], self.models[-1][1], self.models[-1][2], train_cs, iters = baum_welch(self.max_iters, self.models[-1][0], self.models[-1][1], self.models[-1][2], self.observationsT[self.current_fish])
                train_prob = compute_probability(train_cs)
                eprint(train_prob)
                self.train_probabilities.append(train_prob)
                eprint('New Model! ' + str(len(self.models)) + " , Trained for: " + str(iters) + " iterations.")
                self.used_hmm[self.current_fish] = len(self.models)
                return (self.current_fish, len(self.models))
            else:
                eprint('Well, I guess...' + str(idx_max))
                self.used_hmm[self.current_fish] = idx_max
                return (self.current_fish, idx_max)
                
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
        eprint(true_type)
        #self.guesses[] = true_type
        pass
