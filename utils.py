import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from itertools import combinations
from graph import *

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
        nbNodes = int(file.readline().removeprefix("n ").removesuffix("\n"))
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

def generate_convoy(nbVehicules, paths, profits):
    """Attribution des tournées"""
    all_combinations = []
    n = nbVehicules if len(paths)>=nbVehicules else len(paths)
    # Générer toutes les combinaisons de taille 'size'
    for comb in combinations(paths, n):
        # Permuter les chemins dans chaque combinaison
        used_nodes = []
        valid_combination = True
        for path in comb :
            for node in path[1:-1]:
                if node in used_nodes:
                    valid_combination = False
                else : used_nodes.append(node)

        if valid_combination:
            all_combinations.append(comb)

    # Calcul des maxs
    evaluated_combination = []
    for combination in all_combinations :
        evaluated_combination.append((combination,sum(calculate_profit(path,profits)for path in combination)))
    # On tri selon les profits
    evaluated_combination.sort(key=lambda x: x[1], reverse=True)
    if len(evaluated_combination)>0 : return evaluated_combination[0]
    return []