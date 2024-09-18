from math import log
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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

def viterbi_algorithm(A,B,pi,obs):
    B_t = transpose(B)
    delta_past = hadamard_product(pi[0], B_t[obs[0]])
    states = []
    for o in obs[1:]:
        had_matrix = hadamart_matrix(delta_past, A)
        maxs = []
        states_row = []
        for row in had_matrix:
            max_state = max(row)
            previous_state = row.index(max_state)
            maxs.append(max_state)
            states_row.append(previous_state)
        states.append(states_row)

        delta_now = hadamard_product(maxs, B_t[o])
        delta_past = delta_now
    max_val = max(delta_past)
    last_state = delta_past.index(max_val)
    num_rows_states = len(obs)-1
    path = [last_state] + backtrack(states, last_state, num_rows_states-1)
    path.reverse()
    return path

def backtrack(states, current_state, current_row):
    if current_row == -1:
        return []
    else:
        return [states[current_row][current_state]] + backtrack(states, states[current_row][current_state] , current_row-1)
    
def hadamart_matrix(delta,A):
    A_t = transpose(A)
    result = []
    for row in A_t:
        aux = hadamard_product(delta, row)
        result.append(aux)
    return result


'''    
# [A, B, pi], obs = manage_input()
# [A, B, pi], obs = new_manage_input()
maxIters = 100
iters = 0
probs = [-1000000000001, -1000000000000]
while probs[iters] < probs[iters+1] and iters < maxIters:
    iters += 1
    alphas, cs = forward_algorithm(A,B,pi,obs)
    betas = backward_algorithm(A,B,pi,obs,cs)
    di_gammas = compute_di_gamma(alphas, betas, A, B, obs)
    gammas = compute_gamma(di_gammas, alphas)
    [A, B, pi] = estimate_parameters(di_gammas, gammas, obs, B)
    current_prob = compute_probability(cs)
    probs.append(current_prob)
print(manage_output(A,B))
'''
