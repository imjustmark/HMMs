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

# Adapts the input given in Kattis form and returns matrices.
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
    return matrices

# Transposes a matrix.
def transpose(B):
    num_cols_B = len(B[0])
    B_t = []
    for _ in range(0,num_cols_B):
        B_t.append([])
    for row in B:
        for idx, elem in enumerate(row):
            B_t[idx].append(elem)
    return B_t

# Multiplies matrices A and B.
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
    
# Adapts output to fit Kattis format.
def manage_output(res):
    num_rows = len(res)
    num_cols = len(res[0])
    output = str(num_rows) + " " + str(num_cols)
    for elem in res[0]:
        output += " " + str(round(elem, 2))
    return output

[A, B, pi] = manage_input()
state_prediction = matrix_multiplication(pi, A)
observation_prediction = matrix_multiplication(state_prediction, B)
output = manage_output(observation_prediction)
print(output)


        