from AntColony import AntColony
from Utils import *
from itertools import permutations
from sys import maxsize

MATRIX_SIZE = 100


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
    # create_plot(shortest_paths=short_results, title="Hemilton circle search")


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
            # short_results.append(current_pathweight)
            min_path = min(min_path, current_pathweight)

    print("End of the better naive test. It took for -> " + str((time.time() - naive_start_time)) + " seconds")
    print(min_path)
    #create_plot(shortest_paths=short_results, title="Hamiltonian circuit search")


if __name__ == '__main__':
    problem = create_matrix_problem(size=MATRIX_SIZE, min=1, max=100, is_trace_zero=True, is_mirror=True)

    # print(np.matrix.view(problem))

    # run_naive_solution()
    # run_the_better_solution(problem)

    i = 1
    for i in range(4):
        start_time = time.time()
        print("evaporation_rate: " + str(0.8))
        aco_optimize = AntColony(ants_number=((i+1)*50), evaporation_rate=.2, intensification=2, alpha=1, beta=1,
                                 beta_evaporation_rate=0, choose_best=.1)
        best = aco_optimize.fit(problem, max_iterations=1500, stop_count=30, debug=True)
        print("The ACO took for -> " + str(time.time() - start_time) + " seconds")
        print("The best path value is: " + str(best))
        aco_optimize.show_plot()

        start_time = time.time()
        print("evaporation_rate: " + str(0.95))
        aco_optimize = AntColony(ants_number=((i+1)*50), evaporation_rate=.05, intensification=2, alpha=1, beta=1,
                                 beta_evaporation_rate=0, choose_best=.1)
        best = aco_optimize.fit(problem, max_iterations=1500, stop_count=30, debug=True)
        print("The ACO took for -> " + str(time.time() - start_time) + " seconds")
        print("The best path value is: " + str(best))
        aco_optimize.show_plot()

        start_time = time.time()
        print("evaporation_rate: " + str(0.99))
        aco_optimize = AntColony(ants_number=((i+1)*50), evaporation_rate=.01, intensification=2, alpha=1, beta=1,
                                 beta_evaporation_rate=0, choose_best=.1)
        best = aco_optimize.fit(problem, max_iterations=1500, stop_count=30, debug=True)
        print("The ACO took for -> " + str(time.time() - start_time) + " seconds")
        print("The best path value is: " + str(best))
        aco_optimize.show_plot()

        start_time = time.time()
        aco_optimize = AntColony(ants_number=((i+1)*50), evaporation_rate=.2, intensification=2, alpha=1, beta=1,
                                 beta_evaporation_rate=0, choose_best=.1)
        best = aco_optimize.fit(problem, max_iterations=1500, stop_count=30, debug=True)
        print("The ACO took for -> " + str(time.time() - start_time) + " seconds")
        print("The best path value is: " + str(best))

        aco_optimize.show_plot()
