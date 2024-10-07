'''
General information on HMMs:
a_{ij} = Probability of being in state j at t+1 given that at t we are in state i.
b_j (k) = Probability of seeing observation k at t given that we are in state j at t.

Problem 1: Given a model lambda and a sequence of observations O, find P(O|lambda).
Problem 2: Given a model lambda and a sequence of observations O, find the sequence of states X 
which maximizes the probability of obtaining O.
Problem 3: Given a sequence of observations O, compute the model parameters that maximize the probability of obtaining O.

Solution to Problem 1: Forward algorithm. Define alpha_t (i) = P(o_1, o_2, ... , o_t, x_t = i | lambda).
                       Backward algorithm. Define beta_t (i) = P(o_{t+1}, ... o_T | x_t = i, lambda)

Solution to Problem 2: Viterbi algorithm. Define delta_t (i) = Probability of the best path ending at state i in time t given O and lambda.

Solution to Problem 3: Baum-Welch algorithm. Define gamma_t (i,j) = P(x_t = i, x_{t+1} = j | O_{1:T})
                                                    gamma_t (i) = P(x_t = i | O_{1:T})
'''

from math import log
import matplotlib.pyplot as plt
import numpy as np

def manage_input():
    A = []
    B = []
    pi = []
    matrices = [A, B, pi]
    for matrix in matrices:
        input_row = input()
        values = input_row.split(" ")
        num_rows = int(values[0])
        num_cols = int(values[1])
        for idx1 in range(0,num_rows):
            row = []
            for idx2 in range(0, num_cols):
                value = float(values[2+idx2+num_cols*idx1])
                row.append(value)
            matrix.append(row)
    input_row = input()
    values = input_row.split(" ")
    num_obs = int(values[0])
    obs = []
    for idx in range(0,num_obs):
        obs.append(int(values[idx+1]))
    return matrices, obs

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

def forward_algorithm(A,B,pi,obs):
    if len(obs) <= 0:
        return 0
    B_t = transpose(B)
    factor = floor_observation_pdf(B_t[obs[0]]) # First: If all elements of b_t are smaller than our floor, they become our floor.
    alpha = hadamard_product(pi[0], factor)
    if all(x == 0 for x in alpha):              # Second: If all elements of alpha_t are 0, then we stop the computation because we know that sequence of observations is impossible in our model.
        return None, None
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
            return None, None
        alpha, c = normalize(alpha)
        alphas.append(alpha)
        new_cs.append(c)
    return alphas, new_cs

def backward_algorithm(A,B,pi,obs, cs):
    if len(obs) <= 0:
        return 0
    B_t = transpose(B)
    beta = [cs[-1] for _ in pi[0]] # We use the same normalization factors as in alpha so that when reestimating the parameters they cancel each other.
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

# There are only gammas up to T-1 because there is no X_{T+1}.
# Alpha, beta and gamma matrices have shapes Observations x States.
# Di-gamma is a vector of length t with matrices of shape States x States.
def compute_di_gamma(alphas, betas, A, B, obs):
    di_gammas = []
    # norm = sum(alphas[-1])    The sum of alphas[-1] is 1 with our normalisation.
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

# We marginalize across the states at t+1. Gamma is a matrix of shape Observations x States.
# Gamma_T = Alpha_T because its the probability of being at each state in T given O.
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
        if norm == 0:
            norm = 1e-100
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

def new_manage_input():
    f = open("C:/Users/marcd/OneDrive/Escritorio/KTH/AI/Question7.txt", "r")
    A = []
    B = []
    pi = []
    matrices = [A, B, pi]
    for matrix in matrices:
        input_row = f.readline()
        values = input_row.split(" ")
        num_rows = int(values[0])
        num_cols = int(values[1])
        for idx1 in range(0,num_rows):
            row = []
            for idx2 in range(0, num_cols):
                value = float(values[2+idx2+num_cols*idx1])
                row.append(value)
            matrix.append(row)
    input_row = f.readline()
    values = input_row.split(" ")
    num_obs = int(values[0])
    obs = []
    for idx in range(0,num_obs):
        obs.append(int(values[idx+1]))
    return matrices, obs

def compute_probability(cs):
    log_prob = 0
    for c in cs:
        log_prob += log(c)
    log_prob = -log_prob
    return log_prob

def manage_output(A,B):
    output = str(len(A)) + " " + str(len(A[0]))
    for row in A:
        for elem in row:
            output += " " + str(round(elem, 6))
    output += "\n" + str(len(B)) + " " + str(len(B[0]))
    for row in B:
        for elem in row:
            output += " " + str(round(elem, 6))
    return output

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
        
# [A, B, pi], obs = manage_input()
[A, B, pi], obs = new_manage_input()
num_states = 3
'''
A = np.random.beta(2, 2, (num_states, num_states))
B = np.random.beta(2, 2, (num_states, 4))
pi = np.random.beta(2, 2, (1, num_states))
'''
A = np.array([[0.75, 0.03, 0.2],[0.15,0.7,0.2],[0.3,0.3,0.4]])
B = np.array([[0.7, 0.22, 0.08, 0],[0.15,0.4,0.3,0.15],[0.1,0.2,0.2,0.5]])
pi = np.array([[0.8,0.1,0.1]])
sum_A = np.sum(A, axis=1)
A = (A / sum_A[:,np.newaxis]).tolist()
sum_B = np.sum(B, axis=1)
B = (B / sum_B[:,np.newaxis]).tolist()
sum_pi = np.sum(pi, axis=1)
pi = (pi / sum_pi[:,np.newaxis]).tolist()
print(A)
print(B)
print(pi)

# Uniform: 1000: converges in 870 steps to the right value. 10000: 1000 steps to the wrong value.
# Diagonal and pi = [0,0,1] -> Cannot learn anything: Estimates the probability of observations in the only state it can be.
# Matrices close to solution: 10000: Still converges to the wrong value, even when close to the solution. 1000: Converges quickly (472) to the correct solution.
maxIters = 1000
iters = 0
probs = [-1000000000001, -1000000000000]
tol = 1e-5
while (probs[iters] < probs[iters+1]) and (probs[iters+1] - probs[iters] > tol) and iters < maxIters:
    iters += 1
    alphas, cs = forward_algorithm(A,B,pi,obs)
    betas = backward_algorithm(A,B,pi,obs,cs)
    di_gammas = compute_di_gamma(alphas, betas, A, B, obs)
    gammas = compute_gamma(di_gammas, alphas)
    [A, B, pi] = estimate_parameters(di_gammas, gammas, obs, B)
    current_prob = compute_probability(cs)
    probs.append(current_prob)
print(iters)
# print(manage_output(A,B))
print(A)
print(B)
print(pi)
# print(probs[2:])
plt.plot(probs[2:])
plt.show()

# For 10000: N = 2 converges in 270 iters. N = 4 uses up 1000 iters.
# For 1000: N = 2 converges in 550 iters. N = 4 uses up 1000 iters.
