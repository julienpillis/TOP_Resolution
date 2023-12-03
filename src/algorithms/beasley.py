from itertools import combinations
import sys
sys.path.append('../')
import src.ressource.utils as utils
from src.ressource.graph import *
import src.algorithms.localSearch as localS
import src.ressource.TSP_heuristics as tsp
import time

def beasley_top(graph : Graph, heuristic, localSearch):
    """Algorithme de Beasley pour le TOP"""

    # Construction du chemin hamiltonien
    heuristic_path = heuristic(graph.getNodes(),graph.start_point,graph.end_point)

    # On supprime les noeuds de départ et d'arrivée. Ils seront automatiquement ajoutés dans un chemin
    heuristic_path.pop(heuristic_path.index(graph.start_point))
    heuristic_path.pop(heuristic_path.index(graph.end_point))

    paths = []  # Graphe auxiliaire

    for i in range(len(heuristic_path)):
        path = [graph.start_point, graph.end_point]
        j = i
        prev_node_idx = 0
        continue_insertion = True
        while continue_insertion:
            path.insert(prev_node_idx+1,heuristic_path[j])
            duration = utils.calculate_time(path,graph.times)
            if duration <= graph.maxTime :
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

    solution,profits = utils.gen_conv(graph.nbVehicules, paths, graph.profits,graph.nodes)
    if solution == [] : return [],profits
    else :
        # Optimisation des solutions
        profits = 0
        solution = list(solution)
        for i in range(len(solution)):
            used_nodes = utils.extract_inner_tuples(solution)
            better_path= localSearch(solution[i],graph.maxTime,graph.profits,graph.times,graph.nodes,used_nodes)
            if better_path:
                solution[i] = [node for node in better_path]
            profits += utils.calculate_profit(solution[i],graph.profits)

        return solution, profits

def beasley_top_optimized(graph : Graph):
    """Application de toutes les heuristiques du PVC et de recherche locale lors de l'heuristique de Beasley"""
    best_convoy = []
    best_profit = 0
    for heuristic in tsp.heuristics :
        for opt in localS.optimization :
            convoy,profit = beasley_top(graph,heuristic,opt)
            if profit > best_profit :
                best_convoy = convoy
                best_profit = profit
            reversed_convoy,reversed_profit = beasley_top(graph,heuristic,opt)
            if reversed_profit > best_profit:
                best_convoy = [list(reversed(path)) for path in reversed_convoy]
                best_profit = reversed_profit
    return best_convoy,best_profit