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


        