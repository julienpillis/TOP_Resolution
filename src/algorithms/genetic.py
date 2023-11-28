import os
import sys
sys.path.append('..')
from src.ressource.utils import read_file_and_create_graph, drawGraph, Graph

################ LOAD DATA  ####################

# Get the current working directory
current_path = os.getcwd()
# Go up one directory to the parent directory
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
# Construct the path to the 'data' directory
data_path = os.path.join(parent_path, 'data')
# Get a list of all items (files and folders) in the 'data' directory
all_items = os.listdir(data_path)
# Filter the items to include only folders that start with "Set"
set_folders = [item for item in all_items if os.path.isdir(os.path.join(data_path, item)) and item.startswith('Set')]
data_paths=[]
for folder in set_folders:
    data_paths.append(os.path.join(data_path, folder))
directory_path=data_paths[0]
all_items = os.listdir(directory_path)
txt_files = [os.path.join(directory_path, item) for item in all_items if os.path.isfile(os.path.join(directory_path, item)) and item.endswith('.txt')]

################ READ GRAPH ####################
graph_object = read_file_and_create_graph(txt_files[0])

################ GENETIC ALGO ####################
import random

"""
This function groups path by the number of vehicules
"""
def group_population(population,n):
    grouped_list = []
    for i in range(0, len(population), n):
        group = population[i:i+n]
        grouped_list.append(group)
    return grouped_list

class GeneticAlgorithm:
    def __init__(self, graph : Graph, population_size, generations):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()
    """
    This function initializes the population with random solutions
    """
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            route = random.sample(self.graph.getNodes(), len(self.graph.getNodes()))
            population.append(route)
        return group_population(population,self.graph.nbVehicules)
    
    """
    This function repairs a given solution:
    - removes nodes that exceeds the L time limit
    - keeps only unique nodes in a given solution
    """
    def repair_solution(self,population):
        solution=[]
        for route in population:
            total_time = 0
            # we remove start and end to not compare the same nodes
            if self.graph.start_point in route:
                route.remove(self.graph.start_point)
            if self.graph.end_point in route:
                route.remove(self.graph.end_point)
            repaired_route = [self.graph.start_point]
            i=0
            for node in route:
                if total_time + self.graph.getTimes()[(repaired_route[i], node)] <= self.graph.getMaxTime():
                    repaired_route.append(node)
                    total_time += self.graph.getTimes()[(repaired_route[i], node)]
                    i+=1
            while repaired_route and (total_time + self.graph.getTimes()[(repaired_route[-1],self.graph.end_point)])>self.graph.getMaxTime():
                a=repaired_route.pop()
                total_time-=self.graph.getTimes()[(repaired_route[-1],a)]
            repaired_route.append(self.graph.end_point)
            solution.append(repaired_route)
        if len(solution)>0:
            self.unique_nodes(solution)
        return solution
    
    """
    Returns the total profit of a given solution
    """
    def fitness_function(self,convoy):
        total_profit = 0
        for path in convoy:
            total_time = 0
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                profit = self.graph.getProfits()[current_node]
                # Check if adding the next node exceeds the time limit
                if current_node!=next_node and total_time + self.graph.getTimes()[(current_node, next_node)] <= self.graph.getMaxTime():
                    total_profit += profit
                    total_time += self.graph.getTimes()[(current_node, next_node)]
                else:
                    # Return 0 if time limit is exceeded
                    return 0

            # Add profit for the last node (if needed)
            total_profit += self.graph.getProfits()[path[-1]]
        return total_profit
    
    """
    Returns the total duration of a given solution
    """
    def calculate_time(self,convoy):
        total_time = []
        for path in convoy:
            time=0
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                time+=self.graph.getTimes()[(current_node, next_node)]
            total_time.append(time)
        return total_time
    
    """
    Returns the best solution between "tournament_size" sample solutions
    """
    def selection(self, tournament_size):
        candidates = []
        for _ in range(tournament_size):
            candidates.append(random.sample(self.population, tournament_size))
        filtered=[sublist for sublist in candidates[0] if len(sublist)>0]
        selected_candidate = max(filtered, key=self.fitness_function)
        selected_candidate = max(selected_candidate,key=len)
        return selected_candidate

    def mutation(self, convoy):
        mutated_convoy = []
        for path in convoy:
            # Choose a random mutation point for each path
            mutation_point = random.randint(0, len(path) - 1)
            # Use set difference to find nodes that are not in the current path
            remaining_nodes = set(list(self.graph.getNodes())) - set(path)
            # Replace the node at the mutation point with a random remaining node
            mutated_path = path[:mutation_point] + [random.choice(list(remaining_nodes))] + path[mutation_point + 1:]
            #print("mutated=",mutated_path)
            mutated_convoy.append(self.unique_list(mutated_path))
        return mutated_convoy
    
    def crossover(self,parent1, parent2):
        # Initialize the child with an empty list of paths
        child = []
        start_point = parent1[0]
        end_point = parent1[-1]
        p1=parent1[1:-1]
        p2=parent2[1:-1]
        # Iterate through each pair of paths in the parents
        for path1, path2 in zip(p1, p2):
            # Choose a random crossover point for each path
            crossover_point = random.randint(0, min(len(path1), len(path2)))
            # Combine the first part of path1, 
            new_path=[]
            if len(p1[:crossover_point])>0:
                new_path+=(p1[:crossover_point])
            # Crossover part of path2,
            for point in p2:
                if point not in p1:
                    new_path.append(point)
            # And the second part of path1
            if len(p1[crossover_point:])>0:
                new_path+=(p1[crossover_point:])
            
            # Put back the start and end points
            new_path.insert(0, start_point)
            new_path.append(end_point)

            # Ensure that the same tuples are not repeated
            child.append(self.unique_list(new_path))
        return child
    
    """
    Filter the original list to keep only tuples with unique second elements
    """
    def unique_list(self,convoy):
        unique_second_elements = set()
        filtered_list = [item for item in convoy if item[1] not in unique_second_elements and not unique_second_elements.add(item[1])]
        return filtered_list
    
    """
    This function makes the nodes in a given solution unique
    It replaces the replica by a random node in the graph
    """
    def unique_nodes(self,convoy):
        for i,subset in enumerate(convoy):
            for k,node in enumerate(subset[1:-1]):
                for j in range(len(convoy)):
                    if j!=i:
                        if node in convoy[j]:
                            convoy[i][k+1]=random.choice(self.graph.nodes)

    def sorting_key(self, path):
        profit = 0
        for i in range(1,len(path) - 1):
                current_node = path[i]
                profit+=self.graph.getProfits()[current_node]
        sequence_length = len(path)
        return (profit, sequence_length)

    """
    This function applies the steps of a genetic algorithm
    """
    def evolve(self, elitism_ratio=0.1):
        # Calculate the number of individuals to be preserved via elitism
        num_elites = max(1, int(self.population_size * elitism_ratio))
        for generation in range(self.generations):
            new_population = []
            #
            fitness_values = []
            for individual in self.population:
                fitness_values.append((individual, self.fitness_function(individual)))
            sorted_population = sorted(fitness_values, key=lambda x: x[1], reverse=True)
            # Perform elitism by selecting the top solutions
            elites = [individual for individual, _ in sorted_population[:num_elites]]
            new_population.extend(elites)

            # Generate offspring using crossover and mutation
            for _ in range((self.population_size - num_elites) // 2):
                parent1 = self.selection(3)
                parent2 = self.selection(3)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                child1 = self.repair_solution(self.mutation(child1))
                child2 = self.repair_solution(self.mutation(child2))
                new_population.extend([child1, child2])

            self.population = new_population

            best_solution = max(self.population, key=self.fitness_function)
            sorted_elements = sorted(best_solution, key=self.sorting_key, reverse=True)
            # Select the top "nbVehicules" routes
            best_n_elements = sorted_elements[:self.graph.nbVehicules]
            print(f"Generation {generation + 1}, Best Solution: {best_n_elements}, Fitness: {self.fitness_function(best_solution)}", "Time: ",self.calculate_time(best_n_elements))
        return (best_n_elements)
    
ga = GeneticAlgorithm(graph_object, population_size=20*graph_object.nbVehicules, generations=50)
drawGraph(graph=graph_object,solution=ga.evolve())