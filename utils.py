import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from graph import *


def visualize_paths(coordinates, paths, starting_node=None, ending_node=None, profit=-1, exec_time=-1):
    """Visualisation d'une solution du problème TOP"""
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, marker='o', linestyle='', color='b', label='Points')

    for i, path in enumerate(paths):
        path_x = [coord[0] for coord in path]
        path_y = [coord[1] for coord in path]

        plt.plot(path_x, path_y, marker='o', linestyle='-', label=f'Tournée {i + 1}')

    if starting_node is not None:
        plt.plot(starting_node[0], starting_node[-1], marker='o', linestyle='', color='black', label='Noeud de départ')
    if ending_node is not None:
        plt.plot(ending_node[0], ending_node[-1], marker='o', linestyle='', color='grey', label='Noeud d\'arrivée')

    plt.title('Visualisation des Tournées')
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.legend()
    plt.text(0.5, -0.1, f"Profit : {profit}| Execution Time : {exec_time}", ha='center', va='center',
             transform=plt.gca().transAxes,
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
        file.readline()  # On skip la ligne de nombre de noeuds
        nbVehicules = int(file.readline().removeprefix("m ").removesuffix("\n"))
        tmax = float(file.readline().removeprefix("tmax ").removesuffix("\n"))
        nodes = []
        profits = {}

        for line in file:
            x, y, profit = line.split("\t")
            nodes.append((float(x), (float(y))))
            profits[(float(x), float(y))] = int(profit)

    return Graph(nodes, times_calculator(nodes), profits, tmax, nbVehicules)


def times_calculator(points):
    """Calcul des temps entre les points (coût des trajets entre chaque point)"""
    times = {}
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                times[(points[i], points[j])] = distance(points[i], points[j])
    return times


def calculate_profit(path, profits):
    # Calcul du profit total du trajet
    return sum(profits[node] for node in path)


def calculate_time(path, times):
    # Calcul de la durée du trajet
    return sum(times[(path[i], path[i + 1])] for i in range(len(path) - 1))


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


def extract_inner_tuples(array_of_tuples):
    """Extraire les tuples d'une liste de tuples"""
    inner_tuples = []
    for outer_tuple in array_of_tuples:
        for inner_tuple in outer_tuple:
            inner_tuples.append(inner_tuple)
    return inner_tuples
