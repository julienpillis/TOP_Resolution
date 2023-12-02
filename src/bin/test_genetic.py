import os
import time
import sys
import math
sys.path.append('../')
from src.ressource.utils import read_file_and_create_graph, drawGraph, Graph
from src.algorithms.beasley import two_opt
import src.ressource.utils as utils

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
filename=txt_files[0]
print(filename)
graph_object = read_file_and_create_graph(filename)

################ GENETIC ALGO ####################
import random

"""
This function groups path by the number of vehicules
"""

def get_unique_path(path):
    unique_elements_set = []

    for node in path:
        if node in unique_elements_set:
            unique_elements_set.remove(node)
        unique_elements_set.append(node)

    return unique_elements_set

def is_valid_node_convoy(convoy):
        for i,path1 in enumerate(convoy):
            for node in path1[1:-1]:
                for j,path2 in enumerate(convoy):
                    if i!=j:
                        if node in path2[1:-1]:
                            return False
        return True

class GeneticAlgorithm:
    def __init__(self, graph : Graph, population_size, generations):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.population = self.initialize_population()


    def make_nodes_unique(self,convoy):
        unique_nodes = set()
        end_index=len(self.graph.nodes)
        for path in convoy:
            for node in path[1:-1]: 
                if 1 < node < end_index:
                    unique_nodes.add(node)

        for i, path in enumerate(convoy):
            unique_path = [1]  # Start with 1 at the beginning
            for node in path[1:-1]:  # Exclude start and end nodes
                if node in unique_nodes:
                    unique_path.append(node)
                    unique_nodes.remove(node)
            unique_path.append(end_index)  
            convoy[i] = unique_path

        return convoy
    
    def get_profit(self,convoy):
            total_profit = 0
            for path in convoy:
                for i in range(len(path)):
                    current_node = self.graph.nodes[path[i]-1]
                    profit = self.graph.getProfits()[current_node]
                    total_profit += profit
            return total_profit
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            convoy=[]
            for __ in range(self.graph.nbVehicules):
                len_route=random.randint(4,len(self.graph.nodes)//2)
                route=random.sample(range(2, len(self.graph.nodes)-1), len_route)
                route.insert(0,1)
                route.append(len(self.graph.nodes))
                convoy.append(route)
            convoy=self.make_nodes_unique(convoy)
            population.append(convoy)
        return population
    
    def is_valid_time_path(self, path):
        total_time = 0
        for i in range(len(path) - 1):
            current_node = self.graph.nodes[path[i]-1]
            next_node =self.graph.nodes[path[i+1]-1]
            total_time += self.graph.getTimes()[(current_node, next_node)]
        return total_time <= self.graph.getMaxTime()
    
    def fitness_function(self,convoy):
        total_profit = 0
        for path in convoy:
            total_time = 0
            for i in range(len(path) - 1):
                current_node = self.graph.nodes[path[i]-1]
                next_node = self.graph.nodes[path[i+1]-1]
                profit = self.graph.getProfits()[current_node]
                if current_node!=next_node and (total_time + self.graph.getTimes()[(current_node, next_node)] <= self.graph.maxTime):
                    total_profit += profit
                    total_time += self.graph.getTimes()[(current_node, next_node)]
                else:
                    return 0
            last_node=self.graph.nodes[path[-1]-1]
            total_profit += self.graph.getProfits()[last_node]
        #print(total_profit)
        return total_profit
    
    def calculate_time(self,convoy):
        total_time = []
        for path in convoy:
            #print(path)
            start_node = self.graph.nodes[path[0]-1]
            second_node = self.graph.nodes[path[1]-1]
            time=self.graph.getTimes()[(start_node,second_node)]
            for i,index_node in enumerate(path[1:-1]):
                current_node=self.graph.nodes[index_node-1]
                next_node = self.graph.nodes[path[i+2]-1]
                if current_node==next_node:
                    time+=999
                else:
                    time+=self.graph.getTimes()[(current_node, next_node)]
            total_time.append(time)
        return total_time
    
    def aex_crossover(self, parent1, parent2):
        child1 = []
        child2 = []
        end_index=len(self.graph.nodes)
        for path1, path2 in zip(parent1, parent2):
            #print(path1)
            p1 = path1[1:-1]
            p2 = path2[1:-1]
            aex_child1 = self.alternating_edges_crossover(p1, p2)
            aex_child2 = self.alternating_edges_crossover(p2, p1)
            child1.append([1] + aex_child1 + [end_index])
            child2.append([1] + aex_child2 + [end_index])

        return [self.make_nodes_unique(child1), self.make_nodes_unique(child2)]

    def alternating_edges_crossover(self, parent1, parent2):
        child = [parent1[0]]
        visited = set([child[0]])
        current_parent = parent1
        while len(child) < len(parent1):
            next_node = next((node for node in current_parent if node not in visited), None)
            current_parent = parent2 if current_parent == parent1 else parent1
            if next_node:
                child.append(next_node)
                visited.add(next_node)
        return child
    
    def mutation(self, convoy, mutation_probability):
        mutated_convoy = []

        for path in convoy:
            # Introduce a mutation with a dynamically changing probability
            if random.random() < mutation_probability and len(path[1:-1])>2:
                mutated_path = path[1:-1]  # Exclude start and end nodes

                # Randomly select a mutation operator
                mutation_operator = random.choice(['bit_flip', 'random_resetting', 'swap', 'scramble', 'inversion'])

                # Apply the selected mutation operator
                if mutation_operator == 'bit_flip':
                    mutated_path = self.bit_flip_mutation(mutated_path)
                elif mutation_operator == 'random_resetting':
                    mutated_path = self.random_resetting_mutation(mutated_path)
                elif mutation_operator == 'swap':
                    mutated_path = self.swap_mutation(mutated_path)
                elif mutation_operator == 'scramble':
                    mutated_path = self.scramble_mutation(mutated_path)
                elif mutation_operator == 'inversion':
                    mutated_path = self.inversion_mutation(mutated_path)

                # Ensure the mutated path is valid
                #mutated_path = self.correct_solution(mutated_path, convoy)

                # Add the mutated path to the convoy
                mutated_convoy.append([1]+mutated_path+[len(self.graph.nodes)])
            else:
                mutated_convoy.append(path)

        # Ensure the entire convoy is valid
        return self.make_nodes_unique(mutated_convoy)

    def bit_flip_mutation(self, path):
        # Randomly select one or more positions and flip the bits
        positions_to_flip = random.sample(range(len(path)), random.randint(1, len(path)))
        for position in positions_to_flip:
            path[position] = 1 - path[position]  # Assuming binary encoding

        return path

    def random_resetting_mutation(self, path):
        # Randomly choose a gene and assign a random permissible value
        gene_to_reset = random.choice(range(len(path)))
        path[gene_to_reset] = random.choice(range(1,len(self.graph.nodes)-1))# Assuming integer encoding
        return path

    def swap_mutation(self, path):
        # Randomly choose two positions and swap their values
        position1, position2 = random.sample(range(len(path)), 2)
        path[position1], path[position2] = path[position2], path[position1]

        return path

    def scramble_mutation(self, path):
        # Randomly choose a subset of genes and shuffle their values
        subset_size = random.randint(1, len(path))
        subset_indices = random.sample(range(len(path)), subset_size)
        subset_values = [path[i] for i in subset_indices]
        random.shuffle(subset_values)

        for i, index in enumerate(subset_indices):
            path[index] = subset_values[i]

        return path

    def inversion_mutation(self, path):
        # Randomly choose a subset of genes and invert their order
        subset_size = random.randint(1, len(path))
        subset_indices = random.sample(range(len(path)), subset_size)
        subset_values = [path[i] for i in subset_indices]
        subset_values.reverse()

        for i, index in enumerate(subset_indices):
            path[index] = subset_values[i]
        return path
    def delete_useless_solutions(self,convoy):
        for path in convoy:
            if len(path)==2:
                return True
        return False
    
    def convert_index_to_node(self,convoy):
        result=[]
        for path in convoy:
            subset=[]
            for index in path:
                node=self.graph.nodes[index-1]
                subset.append(node)
            result.append(subset)
        return result
    
    def two_opt_convoy(self,convoy):
        profits = 0
        convoy_nodes=self.convert_index_to_node(convoy)
        for i in range(len(convoy_nodes)):
            used_nodes = utils.extract_inner_tuples(convoy_nodes)
            better_path= two_opt(convoy_nodes[i][1:-1],self.graph.maxTime,self.graph.profits,self.graph.times,self.graph.nodes,used_nodes)
            if better_path:
                convoy_nodes[i] = [self.graph.start_point]+[node for node in better_path]+[self.graph.end_point]
            profits += utils.calculate_profit(convoy_nodes[i],self.graph.profits)
        convoy_index=[]
        for path in convoy_nodes:
            subset=[]
            for node in path:
                subset.append(self.graph.nodes.index(node)+1)
            convoy_index.append(subset)
        return convoy_index
    
    def sorting_key(self, convoy):
        len_result = sum(len(path) for path in convoy)
        convoy=self.make_nodes_unique(convoy)
        return (len_result, self.get_profit(convoy), -math.fsum(self.calculate_time(convoy)))

    def evolve(self, elitism_ratio=0.1,mutation_probability=0.8):
        num_elites = max(1, int(self.population_size * elitism_ratio))
        print("TMAX: ",self.graph.maxTime)
        print("NbV: ",self.graph.nbVehicules)
        best_solutions=[]
        #[[],[],[]]
        self.population=[sol for sol in self.population if self.delete_useless_solutions(sol) == False]
        for generation in range(self.generations):
            new_population = []
            fitness_values = []
            for individual in self.population:
                fitness_values.append((individual, self.fitness_function(individual)))
            sorted_population = sorted(fitness_values, key=lambda x: x[1], reverse=True)
            elites = [individual for individual, _ in sorted_population[:num_elites] if self.fitness_function(individual)>0]
            new_population.extend(elites)
            for _ in range((self.population_size - len(elites)) // 2):
                parent1=random.choice(self.population)
                parent2=random.choice(self.population)
                while parent1==parent2:
                    parent2=random.choice(self.population)
                #print(parent1)
                #print(parent2)
                child1,child2=self.aex_crossover(parent1,parent2)
                child1=self.mutation(child1,mutation_probability)
                child2=self.mutation(child2,mutation_probability)
                #
                child1_twOpt=self.two_opt_convoy(self.make_nodes_unique(child1))
                child2_twOpt=self.two_opt_convoy(self.make_nodes_unique(child2))
                #
                new_population.extend([child1])
                new_population.extend([child2])
                new_population.extend([child1_twOpt])
                new_population.extend([child2_twOpt])
                #print(new_population)
                self.population = new_population
                #print(len(self.population))
                self.population=[sol for sol in self.population if self.delete_useless_solutions(sol) == False]
                #print(len(self.population))
            best_solution = max(self.population, key=self.fitness_function)
            #self.population.remove(best_solution)
            #print(best_solutions)
            #print("b " ,best_solution)
            #if best_solution not in best_solutions:
            best_solutions.append(best_solution)
            #print(f"Generation {generation + 1}, Best Solution: {best_solution}, Fitness: {self.get_profit(best_solution)}", "Time: ",self.calculate_time(best_solution))

        #print(best_solutions)
        best_n_elements = max(best_solutions, key=self.sorting_key)
        best_n_elements=self.two_opt_convoy(best_n_elements)
        return self.convert_index_to_node(best_n_elements)

ga = GeneticAlgorithm(graph_object, population_size=250*graph_object.nbVehicules, generations=100)
#ga.evolve()
#parent1=random.choice(ga.population)
start_time = time.time()
result=ga.evolve()
end_time = time.time()
execution_time = end_time - start_time
drawGraph(graph=graph_object,solution=result,filename=filename,execution_time=execution_time)
#print(get_unique_path([ (14.9, 13.2), (11.4, 6.7), (11.7, 20.3), (11.7, 20.3), (10.1, 26.4), (16.3, 13.3)]))