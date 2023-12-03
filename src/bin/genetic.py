import os
import sys
import math
sys.path.append('../')
import src.ressource.utils as utils
from src.ressource.utils import read_file_and_create_graph, drawGraph, Graph
from ressource.TSP_heuristics import farthest_insertion

################ LOAD DATA  ####################

# Get the current working directory
current_path = os.getcwd()
# Go up one directory to the parent directory
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
# Construct the path to the 'data' directory
data_path = os.path.join(parent_path, 'src/data')
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
#and item.startswith('p1.2.a')
################ READ GRAPH ####################
print(txt_files[0])
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

def remove_point_from_route(route, point_to_remove):
    # Check if the point is present in the route
    if point_to_remove in route:
        # If present, remove the point from the route
        route.remove(point_to_remove)

def get_unique_path(path):
    unique_elements_set = []

    for node in path:
        if node in unique_elements_set:
            unique_elements_set.remove(node)
        unique_elements_set.append(node)

    return unique_elements_set

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
            len_route=random.randint(2,len(self.graph.getNodes())-1)
            route = random.sample(self.graph.getNodes(), len_route)
            remove_point_from_route(route,self.graph.start_point)
            remove_point_from_route(route,self.graph.end_point)
            route.insert(0,self.graph.start_point)
            route.append(self.graph.end_point)
            population.append(route)
        group_p=group_population(population,self.graph.nbVehicules)
        group_p=[sol for sol in group_p if self.delete_useless_solutions(sol)==False]
        return group_p
    

    
    def delete_useless_solutions(self,convoy):
        for path in convoy:
            if set(path)=={self.graph.start_point, self.graph.end_point}:
                return True
        return False
        #filtered_list = [sublist for sublist in group_p if set(sublist) != {self.graph.start_point, self.graph.end_point}]
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
            #print("route: ",route)
            for node in route:
                #print("node :",node)
                #print("repaired_route :",repaired_route[i])
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
            solution=self.unique_nodes_convoy(solution)
        return solution
    
    def is_valid_time_path(self, path):
        total_time = 0
        #print(path)
        for i in range(len(path) - 1):
            #print(i)
            current_node = path[i]
            next_node = path[i + 1]
            total_time += self.graph.getTimes()[(current_node, next_node)]

        return total_time <= self.graph.getMaxTime()
    
    def is_valid_node_convoy(self, convoy):
        total_time = 0
        #print(path)
        for i,path1 in enumerate(convoy):
            for node in path1[1:-1]:
                for j,path2 in enumerate(convoy):
                    if i!=j:
                        if node in path2[1:-1]:
                            return False
        return True
    """
    Returns the total profit of a given solution
    """
    def fitness_function(self,convoy):
        total_profit = 0
        for path in convoy:
            total_time = 0
            if len(path)==1:
                path=path[0]
            #print("len(path): ",len(path))
            #print("path: ",path)
            #print("path[0]: ",path[0])
            for i in range(len(path) - 1):
                
                current_node = path[i]
                next_node = path[i + 1]
                #print("current_node: ",current_node)
                #print("next_node: ",next_node)
                profit = self.graph.getProfits()[current_node]
                # Check if adding the next node exceeds the time limit
                if current_node!=next_node and (total_time + self.graph.getTimes()[(current_node, next_node)] <= self.graph.maxTime):
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
        #print(len(convoy[0]))
        #print(convoy)
        for path in convoy:
            #print(path)
            #print("tesssstt ",path)
            time=self.graph.getTimes()[(path[0], path[1])]
            #print(path[1:-1])
            for i,current_node in enumerate(path[1:-1]):
                next_node = path[i + 2]
                #print("CURRENT: ",current_node)
                #print("next_node: ",next_node)
                if current_node==next_node:
                    #print(current_node," ",next_node)
                    time+=999
                else:
                    time+=self.graph.getTimes()[(current_node, next_node)]
            total_time.append(time)
        #print(total_time)
        return total_time
    
    """
    Returns the best solution between "tournament_size" sample solutions
    """
    def selection(self, tournament_size):
        candidates = []
        for _ in range(tournament_size):
            #print(len(self.population))
            candidates.append(random.sample(self.population, tournament_size))
        #print(self.population)
        #print(len(candidates))
        if not candidates:
            return []
        filtered=[sublist for sublist in candidates[0] if len(sublist)>0 and len(sublist)<=self.graph.nbVehicules]
        #print(candidates[0])
        selected_candidate = max(filtered, key=self.fitness_function)
        #print(len(selected_candidate))

        return selected_candidate
    
    def correct_solution(self, path,convoy):
        total_time = 0
        invalid_nodes = []
        fil_path=path[1:-1]
        # Check each pair of nodes and mark invalid ones
        for i,current_node in enumerate(fil_path[:-1]):
            current_node = fil_path[i]
            next_node = fil_path[i + 1]
            total_time += self.graph.getTimes()[(current_node, next_node)]

            if total_time > self.graph.getMaxTime():
                # If the current pair exceeds time constraints, mark the node for removal
                invalid_nodes.append(next_node)

        # Correct the solution by removing the marked invalid nodes
        corrected_path = [node for node in fil_path if node not in invalid_nodes]

        # If the corrected path is empty, add a random node to maintain feasibility
        if not corrected_path:
            r=random.choice(self.graph.getNodes())
            while r in [node  for p in convoy for node in p]:
                    r=random.choice(self.graph.getNodes())
            #print("r: ",r)
            corrected_path.append(r)
            #print(corrected_path)

        return [self.graph.start_point] + corrected_path + [self.graph.end_point]

    def mutation(self, convoy, mutation_probability):
        mutated_convoy = []

        for path in convoy:
            # Introduce a mutation with a dynamically changing probability
            if random.random() < mutation_probability:
                mutated_path = path[1:-1]  # Exclude start and end nodes

                # Randomly choose a mutation type (you can customize this based on your needs)
                mutation_type = random.choice(["shuffle", "swap", "reverse"])

                # Apply the chosen mutation type
                if mutation_type == "shuffle":
                    random.shuffle(mutated_path)
                elif mutation_type == "swap" and len(mutated_path) >= 2:
                    # Swap two random nodes (excluding start and end nodes)
                    index1, index2 = random.sample(range(len(mutated_path)), 2)
                    mutated_path[index1], mutated_path[index2] = mutated_path[index2], mutated_path[index1]
                elif mutation_type == "reverse" and len(mutated_path) >= 2:
                    # Reverse a random subsequence of nodes (excluding start and end nodes)
                    index1, index2 = sorted(random.sample(range(len(mutated_path)), 2))
                    mutated_path[index1:index2+1] = reversed(mutated_path[index1:index2+1])

                # Append start and end nodes
                mutated_path = [self.graph.start_point] + mutated_path + [self.graph.end_point]

                # Ensure uniqueness
                mutated_path = self.unique_nodes_convoy([mutated_path])[0]

                # Check if the mutated path is a valid time path
                if self.is_valid_time_path(mutated_path):
                    mutated_convoy.append(mutated_path)
                else:
                    # If not valid, correct the solution
                    corrected_solution = self.correct_solution(mutated_path, convoy)
                    mutated_convoy.append(corrected_solution)
            else:
                mutated_path = path

        # Filter out invalid solutions
        mutated_convoy = [sol for sol in mutated_convoy if self.is_valid_node_convoy([sol])]

        return mutated_convoy

    
    ## check if nodes already exist in parents
    def crossover(self,parent1, parent2):
        # Initialize the child with an empty list of paths
        child1 = []
        child2 = []
        # Iterate through each pair of paths in the parents
        for path1, path2 in zip(parent1, parent2):
            p1=path1[1:-1]
            p2=path2[1:-1]
            combined_path = p1 + p2
            unique_nodes = list(set(combined_path))

            # Randomly shuffle the unique nodes
            random.shuffle(unique_nodes)

            # Create two children from the shuffled nodes
            split_index = len(unique_nodes) // 2
            sol1=unique_nodes[:split_index]
            sol2=unique_nodes[split_index:]
            #
            sol1.insert(0,self.graph.start_point)
            sol1.append(self.graph.end_point)
            #
            sol2.insert(0,self.graph.start_point)
            sol2.append(self.graph.end_point)
            #
            # Check if the solutions exceed time constraints
            if self.is_valid_time_path(sol1):
                child1.append(sol1)

            if self.is_valid_time_path(sol2):
                child2.append(sol2)
 
        return [child1,child2]
    
    def crossoverrr(self, parent1, parent2):
        # Initialize the children with empty lists of paths
        child1 = []
        child2 = []

        # Iterate through each pair of paths in the parents
        for path1, path2 in zip(parent1, parent2):
            # Extract nodes excluding start and end points
            p1 = path1[1:-1]
            p2 = path2[1:-1]

            # Combine unique nodes from both parents
            combined_path = p1 + p2
            unique_nodes = list(set(combined_path))

            # Ensure unique nodes and shuffle
            random.shuffle(unique_nodes)

            # Create two children from the shuffled nodes
            split_index = len(unique_nodes) // 2
            sol1 = unique_nodes[:split_index]
            sol2 = unique_nodes[split_index:]

            # Insert start and end points
            sol1.insert(0, self.graph.start_point)
            sol1.append(self.graph.end_point)

            sol2.insert(0, self.graph.start_point)
            sol2.append(self.graph.end_point)

            # Check and fix time constraints for child 1
            if not self.is_valid_time_path(sol1):
                sol1 ,b= self.fix_time_constraint(sol1)
                if b:
                    child1.append(sol1)

            # Check and fix time constraints for child 2
            if not self.is_valid_time_path(sol2):
                sol2 ,b= self.fix_time_constraint(sol2)
                if b:
                    child2.append(sol2)

        return [child1, child2]

    def fix_time_constraint(self, path):
        # Implement corrective actions to fix time constraints
        # For example, you can apply mutation or generate a new solution

        # Here, we apply mutation until a valid solution is obtained
        mutated_path = self.mutationnn([path], 1)[0]
        #print(mutated_path)
        i=0
        while (i<32)and (not mutated_path or not self.is_valid_time_path(mutated_path)):
            mutated_path = self.mutationnn([path], 1)[0]
            i+=1
            #print(i)
        return mutated_path,i<=len(self.graph.nodes)
    
    def mutationnn(self, convoy, mutation_probability):
        mutated_convoy = []
        for path in convoy:
            # Introduce a mutation with a dynamically changing probability
            if random.random() < mutation_probability:
                mutated_path = path[1:-1]  # Exclude start and end nodes
                random.shuffle(mutated_path)
                mutated_path = [self.graph.start_point] + mutated_path + [self.graph.end_point]
            else:
                mutated_path = path

            mutated_convoy.append(mutated_path)

        return mutated_convoy

    """
    Filter the original list to keep only tuples with unique second elements
    """
    def unique_nodes_convoy(self,convoy):
        filtered_list=[]
        for path in convoy:
            filtered_list.append(get_unique_path(path))
        return filtered_list

    """
    This function makes the nodes in a given solution unique
    It replaces the replica by a random node in the graph
    """
    def unique_nodes_convert(self,convoy):
        result=convoy
        for i,subset in enumerate(result):
            for k,node in enumerate(subset[1:-1]):
                for j in range(len(result)):
                    if j!=i:
                        if node in result[j]:
                            result[i][k+1]=random.choice(self.graph.nodes)
        return result

    def sorting_key(self, path):
        profit = 0
        for i in range(1,len(path) - 1):
                current_node = path[i]
                profit+=self.graph.getProfits()[current_node]
        sequence_length = len(path)
        return (profit, sequence_length)
    
    def is_valid_solution(self, solution):
        # Nodes constraint
        if not self.is_valid_node_convoy(solution):
            return False
        # Time constraint
        for path in solution:
            if not self.is_valid_time_path(path):
                return False
        return True
    """
    This function applies the steps of a genetic algorithm
    """
    def evolve(self, elitism_ratio=0.1,mutation_probability=0.8):
        # Calculate the number of individuals to be preserved via elitism
        num_elites = max(1, int(self.population_size * elitism_ratio))
        print("TMAX: ",self.graph.maxTime)
        print("NbV: ",self.graph.nbVehicules)
        best_solutions=[]
        for generation in range(self.generations):
            new_population = []
            #
            fitness_values = []
            for individual in self.population:
                fitness_values.append((individual, self.fitness_function(individual)))
            #print(fitness_values)
            sorted_population = sorted(fitness_values, key=lambda x: x[1], reverse=True)
            # Perform elitism by selecting the top solutions
            elites = [individual for individual, _ in sorted_population[:num_elites] if self.fitness_function(individual)>0]
            new_population.extend(elites)
            # Generate offspring using crossover and mutation
            for _ in range((self.population_size - len(elites)) // 2):
                print("i: ",_)
                parent1 = self.unique_nodes_convoy(self.selection(len(new_population)-1))
                parent2 = self.unique_nodes_convoy(self.selection(len(new_population)-1))
                child1,child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1,mutation_probability)
                child2 = self.mutation(child2,mutation_probability)
                #
                if self.is_valid_solution(child1):
                    new_population.extend(child1)
                if self.is_valid_solution(child2):
                    new_population.extend(child2)
                #
                new_population = [sol for sol in new_population if self.delete_useless_solutions(sol) == False]
                self.population = new_population
            best_solution = max(self.population, key=self.fitness_function)
            for sol in best_solution:
                if sol not in best_solutions:
                    best_solutions.append(sol)
        
        sorted_elements = sorted(best_solutions, key=self.sorting_key, reverse=True)
        best_n_elements = sorted_elements[:self.graph.nbVehicules]
        #print(best_solutions)
        print(f"Generation {generation + 1}, Best Solution: {best_n_elements}, Fitness: {self.fitness_function(best_n_elements)}", "Time: ",self.calculate_time(best_n_elements))
        return (best_n_elements)
    
ga = GeneticAlgorithm(graph_object, population_size=50*graph_object.nbVehicules, generations=100)
drawGraph(graph=graph_object,solution=ga.evolve())
#print(get_unique_path([ (14.9, 13.2), (11.4, 6.7), (11.7, 20.3), (11.7, 20.3), (10.1, 26.4), (16.3, 13.3)]))