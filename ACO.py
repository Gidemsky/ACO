import csv

from numpy import genfromtxt

from AntColony import AntColony
from Utils import *
from itertools import permutations
from sys import maxsize

MATRIX_SIZE = 150


def run_naive_solution():
    str_matrix_index = create_matrix_string(size=MATRIX_SIZE)
    short_results = []

    print("Start solving using the naive approach")
    naive_start_time = time.time()
    paths = run_permutation(list(str_matrix_index), 0)

    b_val = float('inf')
    b_path = []

    for path in paths:
        cur_path_val = 0
        j, t_index = 0, 0
        for i in range(len(path)):
            if i == j:
                j += 1
                if j != len(path):
                    t_index = int(path[j])
                else:
                    t_index = int(path[0])
            s_index = int(path[i])
            cur_path_val += problem[s_index, t_index]
            short_results.append(cur_path_val)
        if cur_path_val < b_val:
            b_val = cur_path_val
            b_path = path
    print("End of the naive test. It took for -> " + str((time.time() - naive_start_time)) + " seconds")
    print(b_path)
    print(b_val)
    create_plot(shortest_paths=short_results, title="Hamiltonian circle search")


def run_the_better_solution(problem_matrix):
    arr_problem = create_matrix_arr(MATRIX_SIZE)
    short_results = []
    print("Start solving using the better naive approach")
    naive_start_time = time.time()

    # stores all vertex apart from source vertex
    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    next_permutation = permutations(arr_problem)

    for s in range(MATRIX_SIZE):
        for i in next_permutation:

            # store current Path weight(cost)
            current_pathweight = 0

            # compute current path weight
            k = s
            for j in i:
                current_pathweight += problem_matrix[k][j]
                k = j
            current_pathweight += problem_matrix[k][s]

            # update minimum
            short_results.append(current_pathweight)
            min_path = min(min_path, current_pathweight)

    print("End of the better naive test. It took for -> " + str((time.time() - naive_start_time)) + " seconds")
    print(min_path)
    create_plot(shortest_paths=short_results, title="Hamiltonian circuit search")


def run_aco(ants=400, ev_r=0.05, save_to_file=None, a=4, b=3):
    print("starting the running of the algorithm:\n"
          "ants number: " + str(ants) + "\nevaporation_rate: " + str(ev_r) + "\n")
    aco_optimize = AntColony(ants_number=ants, evaporation_rate=ev_r, intensification=2, alpha=a, beta=b,
                             choose_best=.1)
    best = aco_optimize.fitness(problem, max_iterations=4000, stop_count=30, debug=True)
    print("The best path value is: " + str(best))
    aco_optimize.show_plot(file_name=save_to_file)
    return best


def run_test(mat_problem, runs, ev_r, csv_file, ants=25):
    all_results = []
    print_long_line(title="start the test run")
    for i in range(5):
        best, total_run_time = 0, 0
        for run in range(runs):
            print("\n<- start iteration number " + str(run + 1) + " ->\n")
            start_time = time.time()
            best += run_aco(ants=ants, ev_r=ev_r, save_to_file=csv_file + "_Ant-" + str(ants) + "_Evp-" +
                                                               str(ev_r).replace('.', '') + "_" + str(run + 1))
            total_run_time += time.time() - start_time
        best = best / runs
        total_run_time = (total_run_time / runs) / 60
        all_results.append([ants, ev_r, total_run_time, best])
        ants *= 2
    return all_results


if __name__ == '__main__':

    print("creating the problem to solve\n")
    # problem = create_matrix_problem(size=MATRIX_SIZE, min=1, max=100, is_trace_zero=True, is_mirror=True)
    # np.savetxt("Problem Matrix.csv", problem, fmt='%i', delimiter=",")
    problem = genfromtxt('Problem Matrix.csv', delimiter=',')
    print("the matrix has been created successfully")
    if MATRIX_SIZE <= 50:
        print("the matrix is:\n")
        print(np.matrix.view(problem))
        print("\n")

    # run_the_better_solution(problem.copy())
    # run_naive_solution()

    best_results = [run_test(mat_problem=problem, runs=4, ev_r=.2, csv_file="Test1-")]
    with open("ACO results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ants", "evaporation", "avg time", "best_score"])
        writer.writerows(best_results)
    best_results.append(run_test(mat_problem=problem, runs=4, ev_r=.05, csv_file="Test2-"))
    with open("ACO results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ants", "evaporation", "avg time", "best_score"])
        writer.writerows(best_results)
    # best_results = [run_test(mat_problem=problem, runs=4, ev_r=.01, csv_file="Test3-")]
    best_results.append(run_test(mat_problem=problem, runs=4, ev_r=.01, csv_file="Test3-"))
    with open("ACO results2.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ants", "evaporation", "avg time", "best_score"])
        writer.writerows(best_results)
    best_results.append(run_test(mat_problem=problem, runs=4, ev_r=0, csv_file="Test4-"))
    with open("ACO results2.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ants", "evaporation", "avg time", "best_score"])
        writer.writerows(best_results)
