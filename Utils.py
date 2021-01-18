import numpy as np
import time
import matplotlib.pyplot as plt


def permute(a, l, r, all_path):
    if l == r:
        all_path.append(a.copy())
    else:
        for i in range(l, r + 1):
            a[l], a[i] = a[i], a[l]
            permute(a, l + 1, r, all_path)
            a[l], a[i] = a[i], a[l]


def run_permutation(path_list, param):
    all_path = []
    print("start of the permute")
    start_time = time.time()
    permute(path_list, param, len(path_list) - 1, all_path)
    print("end of the permute. It took " + str(time.time() - start_time) + " seconds")
    return np.array(all_path)


def create_matrix_problem(size, min, max, is_trace_zero, is_mirror):
    matrix = np.random.randint(min, max, size=(size, size))
    if is_mirror:
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                matrix[i, j] = matrix[j, i]
    if is_trace_zero:
        for i in range(size):
            matrix[i, i] = 0
        return matrix
    return matrix


def create_matrix_string(size):
    string = ""
    for i in range(size):
        string += str(i)
    return string


def create_matrix_arr(size):
    index_arr = []
    for i in range(size):
        index_arr.append(i)
    return index_arr


def create_plot(shortest_paths, title):
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(shortest_paths, label="Best Run")
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("The Shortest Path")
    ax.legend()
    plt.title(title)
    plt.show()


def create_and_modify_trace(size, diagonal_src_value=1, diagonal_dst_value=0):
    temp_matrix = np.ones((size, size))
    temp_matrix[np.eye(size) == diagonal_src_value] = diagonal_dst_value
    return temp_matrix, list(range(size))
