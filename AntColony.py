import warnings
import time
import numpy as np

from Utils import create_plot, create_and_modify_trace

warnings.filterwarnings("ignore")


class AntColony:
    def __init__(self, ants_number, evaporation_rate, intensification, alpha=1.0, beta=0.0, choose_best=.1):
        """
        The Ant colony optimizer constructor
        :param ants_number: number of ants to traverse the graph
        :param evaporation_rate: rate that pheromone evaporates
        :param intensification: constant added to the best path
        :param alpha: weighting of pheromone. Must be more or equal to 0
        :param beta: weighting of heuristic (1/distance). Must be more or equal to 1
        :param choose_best: the probability to choose the best route
        """
        # Algorithm parameters
        assert ants_number != 0, "There is no ant in the colony"
        self.ants = ants_number
        self.evaporation_rate = evaporation_rate
        self.pheromone_intensification = intensification

        # init and checks the alpha and beta params
        assert alpha >= 0, "The alpha parameter isn't legal"
        self.alpha = alpha
        assert alpha >= 1, "The beta parameter isn't legal"
        self.beta = beta

        self.choose_best = choose_best

        # algorithm matrices
        self.pheromone_matrix = None
        self.heuristic_matrix = None
        self.probability_matrix = None

        self.problem = None
        self.available_nodes = None

        # Internal stats
        self.best_series = []
        self.best = None
        self.fitted = False
        self.best_path = None
        self.fit_time = None

    def algorithm_initialization(self):
        """
        Initializes the algorithm by creating pheromone, heuristic, probability matrices
        and creating a list of available nodes (cities)
        """
        assert self.problem.shape[0] == self.problem.shape[1], "Map is not a distance matrix!"

        self.pheromone_matrix, self.available_nodes = create_and_modify_trace(size=self.problem.shape[0])
        self.heuristic_matrix = 1 / self.problem
        self.probability_matrix = (self.pheromone_matrix ** self.alpha) * (
                self.heuristic_matrix ** self.beta)

    def next_node(self, located_at_node):
        """
        Chooses the next node to move to according of the probabilities.
        If p < p_choose_best, then the best path is chosen, otherwise
        it is selected from a probability distribution weighted by the pheromone.
        :param located_at_node: the node (city) the ant is located right now
        :return: index of the node the ant is going to
        """
        numerator = self.probability_matrix[located_at_node, self.available_nodes]
        if np.random.random() < self.choose_best:
            next_node = np.argmax(numerator)
        else:
            # calculates the probability according the general function of the kth ant
            p_xy = numerator / np.sum(numerator)
            # get the next node with the highest probabilities
            next_node = np.random.choice(range(len(p_xy)), p=p_xy)
        return next_node

    def evaluate(self, paths):
        """
        Evaluates the solutions of the ants by adding up the distances between nodes.
        :param paths: solutions from the ants
        :return: x and y coordinates of the best path as a tuple, the best path, and the best score
        """
        scores = np.zeros(len(paths))
        coordinates_i, coordinates_j = [], []
        for index, path in enumerate(paths):
            score = 0
            cur_coordinates_i, cur_coordinates_j = [], []
            for i in range(len(path) - 1):
                cur_coordinates_i.append(path[i])
                cur_coordinates_j.append(path[i + 1])
                score += self.problem[path[i], path[i + 1]]
            scores[index] = score
            coordinates_i.append(cur_coordinates_i)
            coordinates_j.append(cur_coordinates_j)
        # saves the ant number (the index) with the smallest path score
        best = np.argmin(scores)
        return (coordinates_i[best], coordinates_j[best]), paths[best], scores[best]

    def update_pheromone(self, best_coordinates):
        """
        The update pheromone works as well:
        1 - evaporate the pheromone
        2 - add pheromone to the best path
        3 - calculate and update the probability matrix
        :param best_coordinates: x and y (i and j) coordinates of the best route
        :return:
        """
        # calculate the pheromone as the inverse of the evaporation rate.
        self.pheromone_matrix *= (1 - self.evaporation_rate)

        # increases the pheromone by a scalar for the best route.
        i = best_coordinates[0]
        j = best_coordinates[1]
        self.pheromone_matrix[i, j] += self.pheromone_intensification

        # the probability updated as followed
        self.probability_matrix = (self.pheromone_matrix ** self.alpha) * (self.heuristic_matrix ** self.beta)

    def fit(self, problem_matrix, max_iterations=100, stop_count=32, debug=False):
        """
        Fits the ACO to the given matrix.
        :param debug: In case we want to see prints of the progress
        :param problem_matrix: Distance matrix or some other matrix with similar properties
        :param max_iterations: number of iterations
        :param stop_count: how many iterations of the same score to make the algorithm stop early
        :return: the best score
        """
        global best_score_so_far
        perv_best_score = None

        if debug:
            print("Start the ACO Optimization with {} iterations...".format(max_iterations))

        start = time.time()
        self.problem = problem_matrix
        self.algorithm_initialization()
        num_equal, num_all_best_equal = 0, 0

        for i in range(max_iterations):
            start_iter = time.time()
            all_path, cur_path = [], []

            for ant in range(self.ants):
                current_node = self.available_nodes[np.random.randint(0, len(self.available_nodes))]
                start_node = current_node
                while True:
                    cur_path.append(current_node)
                    self.available_nodes.remove(current_node)
                    if len(self.available_nodes) == 0:
                        break
                    else:
                        current_node_index = self.next_node(located_at_node=current_node)
                        current_node = self.available_nodes[current_node_index]
                # go back to start
                cur_path.append(start_node)
                # resets the available nodes to all nodes for the next iteration
                self.available_nodes = list(range(self.problem.shape[0]))
                all_path.append(cur_path)
                cur_path = []

            best_path_coordinates, best_path, best_score = self.evaluate(all_path)

            # if perv_best_score is None:
            if perv_best_score == best_score and i != 0:
                num_equal += 1
            else:
                num_equal = 0

            perv_best_score = best_score
            if i == 0:
                best_score_so_far = best_score
            elif best_score < best_score_so_far:
                best_score_so_far = best_score
                self.best_path = best_path

            if best_score == best_score_so_far:
                num_all_best_equal += 1
            else:
                num_all_best_equal = 0

            self.best_series.append(best_score)
            self.update_pheromone(best_coordinates=best_path_coordinates)

            if debug:
                print("Best score at this iteration {}: {}; Best all iterations score: {} ({}s)"
                      "".format(i, round(best_score, 2), round(best_score_so_far, 2),
                                round(time.time() - start_iter)))

            if best_score == best_score_so_far and num_all_best_equal == stop_count:
                print("Stop early due to {} iterations of the same best score.".format(stop_count))
                break

            if num_equal == (stop_count * 3):
                print("Stop early due to {} iterations of the same score.".format(stop_count * 3))
                break

        self.fitted = True
        self.fit_time = round(time.time() - start)
        self.best = self.best_series[np.argmin(self.best_series)]

        if debug:
            print("ACO fitted.  Runtime: {} minutes.  Best score: {}".format(self.fit_time / 60, self.best))

        return self.best

    def show_plot(self, file_name=None):
        """
        Shows the shortest results using plots over time after the algorithm has been fitted.
        :return: None in case the ACO algorithm has not been fitted
        """
        if self.fitted:
            create_plot(shortest_paths=self.best_series, title="Ant Colony Optimization results", file_name=file_name,
                        ants=self.ants, er=self.evaporation_rate, a=self.alpha, b=self.beta, run_time=self.fit_time,
                        best_path=self.best)
        else:
            print("The Ant Colony algorithm has not been fitted!\n")
            return None
