import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from itertools import combinations
from .graph import *
import random
import networkx as nx

import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import matplotlib.patches as mpatches

def drawGraph(graph: Graph, solution,filename="put filename",execution_time="put execution_time"):
    nbCustomers = len(graph.nodes)
    colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(len(solution))]
    plt.figure(figsize=(10, 7))    
    options = {
        'node_size': 100,
        'node_color': 'yellow',
    }

    G = nx.DiGraph()
    G.add_nodes_from(range(nbCustomers))
    pos = {i: (x, y) for i, (x, y) in enumerate(graph.nodes)}

    nx.draw_networkx_nodes(G, pos=pos, **options)
    nx.draw_networkx_labels(G, pos=pos)

    edge_labels = {} 

    for k, path in enumerate(solution):
        mapped_path = [graph.nodes.index(node) for node in path]
        truck_edges = [(mapped_path[i], mapped_path[i + 1]) for i in range(len(mapped_path) - 1)]
        nx.draw_networkx_edges(G, pos=pos, edgelist=truck_edges, edge_color=colors[k], label=f'Path {k + 1}')

        edge_labels.update({(mapped_path[i], mapped_path[i + 1]): f'{round(graph.times[(graph.nodes[i],graph.nodes[i + 1])],1)}' for i in range(len(mapped_path) - 1)})

    nx.draw_networkx_nodes(G, pos=pos, nodelist=[graph.nodes.index(graph.start_point)], node_color='green', node_size=100)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=[graph.nodes.index(graph.end_point)], node_color='red', node_size=100)

    colorbox = [mpatches.Patch(color=colors[k], label=f'Path {k + 1} | Profit : {calculate_profit(solution[k],graph.profits)} | Time : {calculate_time(solution[k],graph.times)}') for k in range(len(solution))]
    
    plt.text(0, -0.1, f'File: {filename}', transform=plt.gca().transAxes)
    plt.text(0, -0.13, f'Execution Time: {execution_time:.2f} seconds', transform=plt.gca().transAxes)
    plt.legend(handles=colorbox, loc='best')

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.show()

def calculate_time_path(path,times):
    total_time = times[(path[0], path[1])]
    for i,current_node in enumerate(path[1:-1]):
        next_node = path[i + 2]
        if current_node==next_node:
            total_time+=0
        else:
            total_time+=times[(current_node, next_node)]
    return round(total_time,3)

def visualize_paths(coordinates, paths, profit, exec_time):
    """Visualisation d'une solution du problème TOP"""
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, marker='o', linestyle='', color='b', label='Points')

    for i, path in enumerate(paths):
        path_x = [coord[0] for coord in path]
        path_y = [coord[1] for coord in path]

        plt.plot(path_x, path_y, marker='o', linestyle='-', label=f'Chemin {i + 1}')

    plt.title('Visualisation des Chemins')
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.legend()
    plt.text(0.5, -0.1, f"Profit : {profit}| Execution Time : {exec_time}", ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True)
    plt.show()


def distance(point1, point2):
    """Calcul de la distance euclidienne entre deux points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def read_file_and_create_graph(file_path):
    """Lecture d'une instance et conversion en objet Graph"""
    with open(file_path, 'r') as file:
        # Ignorer les trois premières lignes
        file.readline() # On skip la ligne de nombre de noeuds
        nbVehicules = int(file.readline().removeprefix("m ").removesuffix("\n"))
        tmax =  float(file.readline().removeprefix("tmax ").removesuffix("\n"))
        nodes = []
        profits = {}

        for line in file:
            x,y,profit = line.split("\t")
            nodes.append((float(x),(float(y))))
            profits[(float(x),float(y))] = int(profit)

    return Graph(nodes,times_calculator(nodes),profits,tmax,nbVehicules)


def times_calculator(points):
    """Calcul des temps entre les points (coût des trajets entre chaque point)"""
    times = {}
    for i in range(len(points)):
        for j in range(len(points)):
            if i!=j :
                times[(points[i],points[j])] = distance(points[i], points[j])
    return times

def calculate_profit(path, profits):
    # Calcul du profit total du trajet
    return sum(profits[node] for node in path)

def calculate_time(path, times):
    # Calcul de la durée du trajet
    return sum(times[(path[i],path[i+1])] for i in range(len(path)-1))

def generate_convoy(nbVehicules, paths, profits, nodes):
    """Attribution des tournées"""
    if not paths:
        return [], -1

    n = nbVehicules if len(paths) >= nbVehicules else len(paths)

    to_study = [path for path in paths]

    # Recherche des chemins avec des noeuds communs
    for node in nodes:
        paths_with_node = []
        # Filtrer les chemins qui contiennent le nœud spécifié
        for path in to_study:
            if node in path[1:-1]:
                paths_with_node.append(path)

        if paths_with_node:
            # Si des chemins avec une même noeud commun ont été trouvés, on ne garde que le meilleur
            chemin_maximal = max(paths_with_node, key=lambda path: calculate_profit(path, profits))
            paths_with_node.remove(chemin_maximal)
            for path in paths_with_node:
                to_study.remove(path)
        # inutile de continuer si tous les chemins ont été étudiés

    to_study.sort(key=lambda path: calculate_profit(path, profits), reverse=True)
    return to_study[:n], sum(calculate_profit(path, profits) for path in to_study[:n])

def gen_conv(nbVehicules, paths, profits, nodes):
    """Attribution des tournées"""
    if not paths:
        return [], -1
    n = nbVehicules if len(paths) >= nbVehicules else len(paths)
    to_study = paths
    #
    nodes_paths={}
    for node in nodes[1:-1]:
        paths_with_node = []
        for path in to_study:
            if node in path[1:-1]:
                paths_with_node.append((calculate_profit(path, profits),path))
        nodes_paths[node]=paths_with_node if len(paths_with_node)>0 else None
    #
    best_solutions=[]
    for node in nodes[1:-1]:
        if nodes_paths[node]:
            _,max_profit_path = max(nodes_paths[node], key=lambda x: x[0])
        
        if max_profit_path in best_solutions:
                continue
        
        best_solutions.append(max_profit_path)
    
    best_solutions=sorted(best_solutions,key=lambda x: calculate_profit(x, profits),reverse=True)
    #
    chosen=[best_solutions[0]]
    for i in range(1,len(best_solutions)):
        exist=False
        for solution in chosen:
            exist = any(node in solution[1:-1] for node in best_solutions[i][1:-1])
            if exist:
                break
        if not exist:
            chosen.append(best_solutions[i])
    #
    n = nbVehicules if len(chosen) >= nbVehicules else len(chosen)
    return chosen[:n], sum(calculate_profit(path, profits) for path in best_solutions[:n])
  
def extract_inner_tuples(array_of_tuples):
    """Extraire les tuples d'une liste de tuples"""
    inner_tuples = []
    for outer_tuple in array_of_tuples:
        for inner_tuple in outer_tuple:
            inner_tuples.append(inner_tuple)
    return inner_tuples