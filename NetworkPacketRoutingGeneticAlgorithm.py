import random

NETWORK_GRAPH = {
    "A": {"B": 4, "C": 2},
    "B": {"A": 4, "C": 1, "D": 5},
    "C": {"A": 2, "B": 1, "D": 8, "E": 10},
    "D": {"B": 5, "C": 8, "E": 2, "Z": 6},
    "E": {"C": 10, "D": 2, "Z": 3},
    "Z": {"D": 6, "E": 3},
}

SOURCE_NODE = "A"
DEST_NODE = "Z"

POPULATION_SIZE = 30
GENERATIONS = 40
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3


def generate_random_path(graph, start, goal, max_hops=10):
    path = [start]
    current = start
    while current != goal and len(path) < max_hops:
        neighbors = [n for n in graph[current].keys() if n not in path]
        if not neighbors:
            return None
        current = random.choice(neighbors)
        path.append(current)
    return path if path[-1] == goal else None


def calculate_latency(graph, path):
    if not path or path[-1] != DEST_NODE:
        return float("inf")
    cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if v not in graph[u]:
            return float("inf")
        cost += graph[u][v]
    return cost


def crossover(p1, p2):
    common_nodes = list(set(p1[1:-1]) & set(p2[1:-1]))
    if not common_nodes:
        return p1[:]

    pivot = random.choice(common_nodes)
    idx1 = p1.index(pivot)
    idx2 = p2.index(pivot)

    offspring = p1[:idx1] + p2[idx2:]

    if len(offspring) == len(set(offspring)):
        return offspring
    return p1[:]


def mutate(graph, path):
    if len(path) <= 2:
        return path

    mutation_point = random.randint(0, len(path) - 2)
    sub_path = path[: mutation_point + 1]
    current = sub_path[-1]

    extension = generate_random_path(graph, current, DEST_NODE)
    if extension:
        candidate = sub_path[:-1] + extension
        if len(candidate) == len(set(candidate)):
            return candidate
    return path


def tournament_selection(population, graph):
    sample = random.sample(population, TOURNAMENT_SIZE)
    sample.sort(key=lambda ind: calculate_latency(graph, ind))
    return sample[0]


def run_genetic_algorithm():
    population = []
    while len(population) < POPULATION_SIZE:
        candidate = generate_random_path(NETWORK_GRAPH, SOURCE_NODE, DEST_NODE)
        if candidate and candidate not in population:
            population.append(candidate)

    best_overall_path = None
    best_overall_cost = float("inf")

    for gen in range(1, GENERATIONS + 1):
        ranked_pop = sorted(
            population, key=lambda p: calculate_latency(NETWORK_GRAPH, p)
        )
        current_best = ranked_pop[0]
        current_cost = calculate_latency(NETWORK_GRAPH, current_best)

        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_path = current_best

        new_population = ranked_pop[:2]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, NETWORK_GRAPH)
            parent2 = tournament_selection(population, NETWORK_GRAPH)

            child = crossover(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child = mutate(NETWORK_GRAPH, child)

            if child and child[-1] == DEST_NODE:
                new_population.append(child)
            else:
                new_population.append(parent1)

        population = new_population

    return best_overall_path, best_overall_cost


best_path, total_cost = run_genetic_algorithm()
print(f"Optimal Packet Route: {' -> '.join(best_path)}")
print(f"Total Path Latency : {total_cost} ms")