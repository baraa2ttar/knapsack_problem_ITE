from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# Genetic Algorithm for solving the Knapsack Problem
def genetic_algorithm(weights, values, capacity, population_size=10, generations=100, mutation_rate=0.1):
    """
    Genetic Algorithm for solving the knapsack problem.

    Parameters:
        weights (list): List of weights for items.
        values (list): List of values for items.
        capacity (int): Maximum weight capacity of the knapsack.
        population_size (int): Number of solutions in the population.
        generations (int): Number of iterations for the algorithm.
        mutation_rate (float): Probability of mutation.

    Returns:
        tuple: Best solution, its total value, and total weight.
    """
    
    # Initialize population with random solutions
    def initialize_population():
        return [[random.choice([0, 1]) for _ in range(len(weights))] for _ in range(population_size)]

    # Fitness function to evaluate solutions
    def fitness(individual):
        total_value = sum(v * i for v, i in zip(values, individual))
        total_weight = sum(w * i for w, i in zip(weights, individual))
        return total_value if total_weight <= capacity else 0  # Penalize invalid solutions

    # Selection process: Pick top 2 solutions based on fitness
    def select(population):
        return sorted(population, key=fitness, reverse=True)[:2]

    # Crossover function: Combine two parents to produce two children
    def crossover(parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    # Mutation function: Randomly flip bits in a solution
    def mutate(individual):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    # Initialize the population
    population = initialize_population()

    # Run the genetic algorithm for the specified number of generations
    for _ in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select(population)  # Select top solutions
            child1, child2 = crossover(parent1, parent2)  # Perform crossover
            new_population.append(mutate(child1))  # Apply mutation
            new_population.append(mutate(child2))  # Apply mutation
        population = new_population

    # Select the best solution from the final population
    best_solution = max(population, key=fitness)
    total_weight = sum(w * i for w, i in zip(weights, best_solution))
    max_value = fitness(best_solution)
    return best_solution, max_value, total_weight

@app.route("/solve", methods=["POST"])
def solve_knapsack():
    """
    Endpoint to solve the knapsack problem using the genetic algorithm.

    Expects a JSON payload with:
        - weights: List of item weights.
        - values: List of item values.
        - capacity: Maximum capacity of the knapsack.
        - num_items (optional): Number of items to consider.

    Returns:
        JSON response with the selected items, maximum value, and total weight.
    """
    data = request.get_json()

    # Parse input data
    num_items = int(data.get("num_items", len(data["weights"])))  # Default to all items if not provided
    weights = list(map(int, data["weights"]))[:num_items]
    values = list(map(int, data["values"]))[:num_items]
    capacity = int(data["capacity"])

    # Solve the knapsack problem using the genetic algorithm
    solution, max_value, total_weight = genetic_algorithm(weights, values, capacity)

    # Prepare the response
    selected_items = [i + 1 for i, selected in enumerate(solution) if selected == 1]
    return jsonify({
        "selected_items": selected_items,
        "max_value": max_value,
        "total_weight": total_weight,
    })

if __name__ == "__main__":
    app.run(debug=True)
