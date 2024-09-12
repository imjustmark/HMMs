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
    num_rows_A = len(A)
    num_cols_A = len(A[0])
    num_rows_B = len(B)
    if num_rows_B != num_cols_A:
        print("NOT MULTIPLIABLE!!!!")
        return
    else:
        res = []
        for _ in range(0,num_rows_A):
            res.append([])

        B_t = transpose(B)
        for i, rowA in enumerate(A):
            for j, rowB in enumerate(B_t):
                sum = 0
                for idx in range(0,num_cols_A):
                    valA = rowA[idx]
                    valB = rowB[idx]
                    sum += valA*valB
                res[i].append(sum)
        return res

def hadamard_product(A,B):
    result = []
    for i in range(0,len(A)):
        result.append(A[i]*B[i])
    return result

def hadamart_matrix(delta,A):
    A_t = transpose(A)
    result = []
    for row in A_t:
        aux = hadamard_product(delta, row)
        result.append(aux)
    return result

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
    
def manage_output(res):
    output = str(res[0])
    for elem in res[1:]:
        output += " " + str(elem)
    return output

"""
matrix = [[1,0,0],[0,1,2],[1,2,1]]
current_state = 2
current_row = 2
path = backtrack(matrix, current_state, current_row)
print(path)
"""

[A, B, pi], obs = manage_input()
path = viterbi_algorithm(A,B,pi,obs)
print(manage_output(path))


        