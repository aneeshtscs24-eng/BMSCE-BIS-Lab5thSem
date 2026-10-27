import numpy as np
class AntColonyRouting:
    def __init__(
        self,
        distance_matrix,
        num_ants,
        num_iterations,
        decay,
        alpha=1.0,
        beta=2.0,
    ):
        self.distances = distance_matrix
        self.num_nodes = len(distance_matrix)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.eta = 1.0 / np.where(
            distance_matrix == 0, np.inf, distance_matrix
        )
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))
    def run(self, source, destination):
        best_path = None
        best_cost = float("inf")
        for iteration in range(self.num_iterations):
            paths, costs = self._generate_ant_paths(source, destination)
            self._update_pheromones(paths, costs)
            if len(costs) > 0:
                min_cost_idx = np.argmin(costs)
                if costs[min_cost_idx] < best_cost:
                    best_cost = costs[min_cost_idx]
                    best_path = paths[min_cost_idx]

        return best_path, best_cost
    def _generate_ant_paths(self, source, destination):
        paths = []
        costs = []
        for ant in range(self.num_ants):
            path = [source]
            visited = set([source])
            current_node = source
            cost = 0.0
            while current_node != destination:
                probabilities = self._calculate_probabilities(
                    current_node, visited
                )
                if probabilities is None:
                    break  
                next_node = np.random.choice(
                    range(self.num_nodes), p=probabilities
                )
                path.append(next_node)
                visited.add(next_node)
                cost += self.distances[current_node][next_node]
                current_node = next_node
            if current_node == destination:
                paths.append(path)
                costs.append(cost)
        return paths, costs
    def _calculate_probabilities(self, current_node, visited):
        unvisited = [
            node
            for node in range(self.num_nodes)
            if node not in visited and self.distances[current_node][node] > 0
        ]
        if not unvisited:
            return None
        pheromones = np.array(
            [
                self.pheromone[current_node][node] ** self.alpha
                for node in unvisited
            ]
        )
        heuristics = np.array(
            [self.eta[current_node][node] ** self.beta for node in unvisited]
        )
        probs = pheromones * heuristics
        sum_probs = np.sum(probs)
        if sum_probs == 0:
            probs = np.ones(len(unvisited)) / len(unvisited)
        else:
            probs = probs / sum_probs
        full_probabilities = np.zeros(self.num_nodes)
        for i, node in enumerate(unvisited):
            full_probabilities[node] = probs[i]
        return full_probabilities
    def _update_pheromones(self, paths, costs):
        self.pheromone *= 1 - self.decay
        for path, cost in zip(paths, costs):
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.pheromone[u][v] += 1.0 / cost
                self.pheromone[v][u] += 1.0 / cost
network_graph = np.array(
    [
        [0, 10, 15, 0, 0],  
        [10, 0, 35, 25, 0],  
        [15, 35, 0, 30, 20],  
        [0, 25, 30, 0, 5],  
        [0, 0, 20, 5, 0],  
    ]
)
aco = AntColonyRouting(
    distance_matrix=network_graph,
    num_ants=20,
    num_iterations=50,
    decay=0.1,
    alpha=1.0,
    beta=2.0,
)
best_route, total_latency = aco.run(source=0, destination=4)
print(f"Optimal Network Route: {best_route}")
print(f"Total Latency/Cost: {total_latency} ms")