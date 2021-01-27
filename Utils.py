import numpy as np
import time
import matplotlib.pyplot as plt


def permute(a, l, r, all_path):
    """
    the permutation
    :param a: the list
    :param l:
    :param r: the length - 1 of the list
    :param all_path: an empty list for all the permutations
    :return:
    """
    if l == r:
        all_path.append(a.copy())
    else:
        for i in range(l, r + 1):
            a[l], a[i] = a[i], a[l]
            permute(a, l + 1, r, all_path)
            a[l], a[i] = a[i], a[l]


def run_permutation(path_list, param):
    """
    permutation calculation of the array
    :param path_list:
    :param param:
    :return:
    """
    all_path = []
    print("start of the permute")
    start_time = time.time()
    permute(path_list, param, len(path_list) - 1, all_path)
    print("end of the permute. It took " + str(time.time() - start_time) + " seconds")
    return np.array(all_path)


def create_matrix_problem(size, min, max, is_trace_zero, is_mirror):
    """
    create the matrix to the tsp problem format
    :param size:
    :param min:
    :param max:
    :param is_trace_zero:
    :param is_mirror:
    :return:
    """
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
    """
    the matrix convert to string
    :param size:
    :return:
    """
    string = ""
    for i in range(size):
        string += str(i)
    return string


def create_matrix_arr(size):
    """
    the problem matrix to array
    :param size:
    :return:
    """
    index_arr = []
    for i in range(size):
        index_arr.append(i)
    return index_arr


def create_plot(shortest_paths, title, file_name=None, ants=None, er=None, a=None, b=None, run_time=None,
                best_path=None):
    """
    the plot creation function.
    all the values are the value to be printed if there are exist
    :param shortest_paths:
    :param title:
    :param file_name:
    :param ants:
    :param er:
    :param a:
    :param b:
    :param run_time:
    :param best_path:
    :return:
    """
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.plot(shortest_paths, label="Best Run")
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("The Shortest Path")
    ax.text(.8, .6,
            'summary:\n'
            'Ants in colony: {}\nEvaporation Rate: {}\nAlpha: {}\nBeta: {}\n\nRunning Time: {} minute\n'
            '\nBest run: {}'.format(
                ants, er, a, b, run_time // 60, best_path),
            bbox={'facecolor': 'gray', 'alpha': 0.8, 'pad': 10}, transform=ax.transAxes)
    ax.legend()
    plt.title(title + " ")
    if not file_name:
        plt.title(title)
        plt.show()
    else:
        plt.title(title + " - " + file_name + ".png")
        plt.savefig(file_name)


def create_and_modify_trace(size, diagonal_src_value=1, diagonal_dst_value=0):
    """
    set the trace to to 0
    :param size:
    :param diagonal_src_value:
    :param diagonal_dst_value:
    :return:
    """
    temp_matrix = np.ones((size, size))
    temp_matrix[np.eye(size) == diagonal_src_value] = diagonal_dst_value
    return temp_matrix, list(range(size))


def print_long_line(title):
    """
    line separator
    :param title: the title to be print in the center of the line separator
    :return:
    """
    print("\n<-------------- " + title + " -------------->\n")
