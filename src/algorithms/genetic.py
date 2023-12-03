import os
import time
import sys
import math
sys.path.append('../')
from src.ressource.utils import read_file_and_create_graph, drawGraph, Graph
from localSearch import two_opt,three_opt
import src.ressource.utils as utils
from src.ressource.TSP_heuristics import farthest_insertion
from itertools import permutations

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
directory_path=data_paths[1]
all_items = os.listdir(directory_path)
txt_files = [os.path.join(directory_path, item) for item in all_items if os.path.isfile(os.path.join(directory_path, item)) and item.startswith('p5.4.w.txt') and item.endswith('.txt')]
#and item.startswith('p5.4.w.txt')

################ READ GRAPH ####################
filename=txt_files[0]
print(filename)
graph_object = read_file_and_create_graph(filename)

################ GENETIC ALGO ####################
import random

"""
This function groups path by the number of vehicules
"""

class GeneticAlgorithm:
    def __init__(self, graph : Graph, population_size, generations):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.population = self.init_population()
        self.bad_nodes=self.get_bad_nodes()

    def get_bad_nodes(self):
        # On supprime les noeuds inatteignables
        used_nodes = [] 
        for point in self.graph.nodes :
            if point not in [self.graph.start_point,self.graph.end_point] and self.graph.times[(self.graph.start_point,point)] + self.graph.times[(point,self.graph.end_point)] > self.graph.maxTime :
                used_nodes.append(point)
        return used_nodes
    
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

    def make_nodes_unique_path(self,path):
        unique_nodes = [path[0]]
        end_index=len(self.graph.nodes)
        for node in path[1:-1]: 
            if (node not in [1,end_index,self.graph.start_point,self.graph.end_point]) and ( node not in unique_nodes ) :
                unique_nodes.append(node)
        unique_nodes.append(path[-1])
        return unique_nodes
    
    def get_profit(self,path):
            total_profit = 0
            for i in range(len(path)):
                current_node = self.graph.nodes[path[i]-1]
                profit = self.graph.getProfits()[current_node]
                total_profit += profit
            return total_profit
    
    def init_population(self):
        start=self.graph.start_point
        end=self.graph.end_point
        heuristic_path = farthest_insertion(self.graph.getNodes(),start,end)
        paths = [] # Sauvegarde les chemins qui sont compatibles (temps <tmax)
        # On supprime les noeuds de départ et d'arrivée. Ils seront automatiquement ajoutés dans un chemin
        heuristic_path.pop(heuristic_path.index(start))
        heuristic_path.pop(heuristic_path.index(end))
        index_path=[]
        for node in heuristic_path:
            index_path.append(self.graph.nodes.index(node)+1)
        heuristic_path=index_path
        for i in range(len(heuristic_path)):
            path = [1, len(self.graph.nodes)]
            j = i
            prev_node_idx = 0
            continue_insertion = True
            while continue_insertion:
                path.insert(prev_node_idx+1,heuristic_path[j])
                duration = utils.calculate_time(self.convert_index_to_node(path),self.graph.times)
                if duration <= self.graph.maxTime :
                    paths.append([node for node in path])
                    # Si le chemin passant par ce noeud ne dépasse pas tmax, on peut l'ajouter au convoi si son profit est meilleur que les autres chemins
                    prev_node_idx += 1
                    j+=1
                    if j>=len(heuristic_path):
                        # Si on a déjà étudié le dernier noeud, on s'arrête
                        continue_insertion = False
                else :
                    # Si tmax dépassé, on ne tente plus d'insertion
                    continue_insertion = False
        return paths

    def fitness_function(self,path):
        total_profit = 0
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
        return total_profit
    
    def calculate_time(self,path):
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
        return time
    
    def aex_crossover(self, path1, path2):
        end_index = len(self.graph.nodes)

        p1 = path1[1:-1]
        p2 = path2[1:-1]
        
        # Perform alternating edges crossover
        aex_child1 = self.alternating_edges_crossover(p1, p2)
        aex_child2 = self.alternating_edges_crossover(p2, p1)

        # Ensure time limit and node uniqueness constraint
        child1 = self.make_nodes_unique_path([1] + aex_child1 + [end_index])
        child2 = self.make_nodes_unique_path([1] + aex_child2 + [end_index])

        return child1, child2
    
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
    
    def mutation(self, path, mutation_probability,mutation_operator=None):
        # Introduce a mutation with a dynamically changing probability
        if random.random() < mutation_probability and len(path[1:-1])>2:
            mutated_path = path[1:-1]  # Exclude start and end nodes

            # Randomly select a mutation operator
            if not mutation_operator:
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
            mutated_path=[1]+mutated_path+[len(self.graph.nodes)]

            # Add the mutated path to the convoy
            return self.make_nodes_unique_path(mutated_path)
        else:
            return self.make_nodes_unique_path(path)


    def bit_flip_mutation(self, path):
        # Randomly select one or more positions and flip the bits
        positions_to_flip = random.sample(range(1, len(path) - 1), random.randint(1, len(path) - 2))
        
        for position in positions_to_flip:
            # Ensure the new node is unique in the convoy
            unique_node = random.choice([node for node in range(2, len(self.graph.nodes)) if node not in path[1:-1]])
            path[position] = unique_node

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
    
    def delete_useless_solutions(self,path):
        return len(path)<3
    
    def convert_index_to_node(self,path):
        subset=[]
        for index in path:
            node=self.graph.nodes[index-1]
            subset.append(node)
        return subset
    
    def convert_convoy_index_to_node(self,convoy):
        result=[]
        for path in convoy:
            subset=[]
            for index in path:
                node=self.graph.nodes[index-1]
                subset.append(node)
            result.append(subset)
        return result
    
    def sorting_key(self, path):
        return (len(path), self.get_profit(path), -self.calculate_time(path))
    
    def two_opt_convoy(self,convoy_nodes):
        new_convoy=[]
        for path in convoy_nodes:
            path=self.make_nodes_unique_path(path)
            two_opt_sol=two_opt(path[1:-1],self.graph.maxTime,self.graph.profits,self.graph.times,self.graph.nodes,self.bad_nodes)
            better_path= two_opt_sol
            new_path=path
            if utils.calculate_profit([self.graph.start_point]+better_path+[self.graph.end_point],self.graph.profits)>utils.calculate_profit(path,self.graph.profits):
                new_path=[self.graph.start_point]+better_path+[self.graph.end_point]
            new_convoy.append(new_path)
        return new_convoy
    
    def three_opt_convoy(self,convoy_nodes):
        new_convoy=[]
        for path in convoy_nodes:
            path=self.make_nodes_unique_path(path)
            three_opt_sol=three_opt(path[1:-1],self.graph.maxTime,self.graph.profits,self.graph.times,self.graph.nodes,self.bad_nodes)
            better_path= three_opt_sol
            new_path=path
            if utils.calculate_profit([self.graph.start_point]+better_path+[self.graph.end_point],self.graph.profits)>utils.calculate_profit(path,self.graph.profits):
                new_path=[self.graph.start_point]+better_path+[self.graph.end_point]
            new_convoy.append(new_path)
        return new_convoy
    
    def calculate_profit_convoy(self,convoy):
        profit=0
        for path in convoy:
            profit+=utils.calculate_profit(path,self.graph.profits)
        return profit
        
    def evolve(self, elitism_ratio=0.1,mutation_probability=0.8):
        num_elites = max(1, int(self.population_size * elitism_ratio))
        print("TMAX: ",self.graph.maxTime)
        print("NbV: ",self.graph.nbVehicules)
        self.population=[sol for sol in self.population if self.delete_useless_solutions(sol) == False]
        for generation in range(self.generations):
            print(f"Generation {generation + 1}")
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
                child1_co,child2_co=self.aex_crossover(parent1,parent2)
                child1=self.mutation(child1_co,mutation_probability)
                child2=self.mutation(child2_co,mutation_probability)
                if self.fitness_function(child1)>0:
                    new_population.append(child1)
                if self.fitness_function(child2)>0:
                    new_population.append(child2)
                self.population.extend(new_population)
                self.population=[sol for sol in self.population if self.delete_useless_solutions(sol) == False]

        random.shuffle(self.population)
        result,_=utils.gen_conv(self.graph.nbVehicules,self.convert_convoy_index_to_node(self.population),self.graph.profits,self.graph.nodes)
        result=max(self.three_opt_convoy(result),self.two_opt_convoy(result),key=self.calculate_profit_convoy)
        return result
    

    
ga = GeneticAlgorithm(graph_object, population_size=100*graph_object.nbVehicules, generations=5)

start_time = time.time()
result=ga.evolve()
end_time = time.time()
execution_time = end_time - start_time
drawGraph(graph=graph_object,solution=result,filename=filename,execution_time=execution_time)
