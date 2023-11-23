from itertools import combinations
import utils
from graph import *


def beasley_top(graph : Graph, starting_point : (int,int), ending_point : (int,int), tmax : int, nbVehicules :int, heuristic = farthest_insertion):
    """Algorithme de Beasley pour le TOP"""
    heuristic_path = heuristic(graph.getNodes(),starting_point,ending_point)
    convoy = []
    paths = []
    # On supprime les noeuds de départ et d'arrivée. Ils seront automatiquement ajouté dans un chemin
    heuristic_path.pop(heuristic_path.index(starting_point))
    heuristic_path.pop(heuristic_path.index(ending_point))
    for i in range(len(heuristic_path)):
        path = [starting_point, ending_point]
        j = i
        prev_node_idx = 0
        continue_insertion = True
        while continue_insertion:
            path.insert(prev_node_idx+1,heuristic_path[j])
            #print(path,calculate_profit(path,graph.profits),calculate_time(path,graph.times))
            duration = utils.calculate_time(path,graph.times)
            if duration <= tmax :
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
    solution,profits = utils.generate_convoy(nbVehicules, paths, graph.profits)
    print("Sum Time before 2-Opt :", sum([utils.calculate_time(solution[i], graph.times) for i in range(len(solution))]))
    print("Sum Profits before 2-Opt :", profits)
    profits = 0
    solution = list(solution)
    for i in range(len(solution)):
        better_path= two_opt(solution[i],graph.maxTime,graph.profits,graph.times,graph.nodes)
        if better_path != []:
            solution[i] = [node for node in better_path]
        profits += utils.calculate_profit(solution[i],graph.profits)

    print("Sum Time after 2-Opt :", sum([utils.calculate_time(solution[i], graph.times) for i in range(len(solution))]))
    print("Sum Profits after 2-Opt :", profits)
    if solution == [] : return [],profits
    else : return solution,profits

def two_opt(path,tmax, profits,times,nodes):
    delta_min = float("inf")

    # Génération des couples de noeuds
    edges = []
    better_path = []
    better_time = utils.calculate_time(path,times)
    #better_profit = calculate_profit(path,profits)
    for i in range(len(path) - 1):
        edges.append((path[i], path[i+1]))

    for edge1,edge2 in combinations(edges,2) :
        # Si la première arête est après la deuxième ou si elles sont contigues, on ne respecte pas le critère
        if path.index(edge1[0])>=path.index(edge2[0]) or path.index(edge1[0])==path.index(edge2[0])+1  :
            break
        else :
            # Croisement sur chemin temporaire
            tmp_path = [node for node in path]

            edge_idx_first = tmp_path.index(edge1[1])
            edge_idx_second = tmp_path.index(edge2[0])
            tmp_path[edge_idx_first],tmp_path[edge_idx_second] = tmp_path[edge_idx_second],tmp_path[edge_idx_first]

            #tmp_profit = calculate_profit(tmp_path,profits)
            tmp_time = utils.calculate_time(tmp_path,times)
            # Si la durée du nouveau chemin est améliorée, on enregistre le chemin
            if tmp_time<better_time : #and tmp_profit>better_profit:
                # Sauvegarde du chemin en comme un meilleur chemin
                better_path = [node for node in tmp_path]
                better_time= tmp_time
                #better_profit = tmp_profit
    if better_path != []:
        insert_nearest_nodes(nodes,better_path, tmax, times)
    return better_path

def insert_nearest_nodes(coords, path , tmax ,times):
    """Recherche des points les plus proches du chemin"""
    node_inserted = False
    min = float("inf")
    for i in range(len(path) - 1):
        for new_node in coords:
            if new_node not in path:
                # Calcul de la distance du noeud par rapport à chaque arête du chemin
                distance_to_node = utils.distance(path[i], new_node) + utils.distance(new_node, path[i + 1])
                if distance_to_node <= tmax - utils.calculate_time(path,times) and distance_to_node < min:
                    # Si l'ajout du noeud ne fait pas dépasser la durée maximale du trajet, on peut l'ajouter
                    nearest_node = new_node
                    idx_to_be_placed = path.index(path[i])+1
                    node_inserted = True
    if node_inserted :
        path.insert(idx_to_be_placed,nearest_node)

    return node_inserted




