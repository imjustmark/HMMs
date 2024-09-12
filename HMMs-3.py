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
    return res, c

def forward_algorithm(A,B,pi,obs):
    if len(obs) <= 0:
        return 0
    B_t = transpose(B)
    alpha_past = hadamard_product(pi[0], B_t[obs[0]])
    alphas = [alpha_past]
    alpha_past, c_past = normalize(alpha_past)
    for o in obs[1:]:
        alpha_now = hadamard_product(matrix_multiplication(alpha_past, A), B_t[o])
        aux_alpha = []
        for elem in alpha_now:
            aux_alpha.append(c_past * elem)
        alphas.append(aux_alpha)
        alpha_now, c_now = normalize(aux_alpha)
        alpha_past = alpha_now
        c_past = c_now
    return alphas, sum(alphas[-1])

def backward_algorithm(A,B,pi,obs):
    if len(obs) <= 0:
        return 0
    B_t = transpose(B)
    beta_past = [1 for _ in pi[0]]
    betas = [beta_past]
    beta_past, c_past = normalize(beta_past)
    new_obs = obs.copy()
    new_obs.reverse()
    for o in new_obs:
        beta_now = matrix_multiplication(hadamard_product(beta_past, B_t[o]), transpose(A))
        beta_aux = []
        for elem in beta_now:
            beta_aux.append(c_past * elem)
        betas.append(beta_aux)
        beta_now, c_now = normalize(beta_aux)
        beta_past = beta_now
        c_past = c_now
    return betas, sum(betas[-1])
"""
def compute_di_gamma(alphas, betas, A, B):
    di_gammas = []
    for t in len(alphas):
"""

[A, B, pi], obs = manage_input()
prob, res_sum = backward_algorithm(A,B,pi,obs)
prob2, res_sum2 = forward_algorithm(A,B,pi,obs)
print(len(prob))
print(prob)
print(len(prob2))
print(prob2)
print(res_sum)
print(res_sum2)