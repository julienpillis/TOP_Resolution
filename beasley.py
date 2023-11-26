from itertools import combinations
import utils
from graph import *


def beasley_top(graph : Graph, starting_point : (int,int), ending_point : (int,int), tmax : int, nbVehicules :int, heuristic):
    """Algorithme de Beasley pour le TOP"""
    heuristic_path = heuristic(graph.getNodes(),starting_point,ending_point)
    convoy = []
    paths = [] # Garde les chemins qui compatibles (temps <tmax)

    #utils.visualize_paths(graph.nodes, [heuristic_path], 0, 0)
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
    #utils.visualize_paths(graph.nodes, [path for path in paths], 0, 0)
    solution,profits = utils.generate_convoy(nbVehicules, paths, graph.profits,graph.nodes)
    if solution == [] : return [],profits
    else :
        print("Sum Profits before 2-Opt :", profits)
        profits = 0
        solution = list(solution)
        for i in range(len(solution)):
            used_nodes = utils.extract_inner_tuples(solution)
            print("(Before 2-Opt) Time path n°", i, "=", utils.calculate_time(solution[i], graph.times))
            better_path= two_opt(solution[i],graph.maxTime,graph.profits,graph.times,graph.nodes,used_nodes)
            if better_path != []:
                solution[i] = [node for node in better_path]
            profits += utils.calculate_profit(solution[i],graph.profits)
            print("(After 2-Opt) Time path n°",i,"=",utils.calculate_time(solution[i],graph.times))

        print("Sum Profits after 2-Opt :", profits)
        return solution, profits

def two_opt(path,tmax, profits,times,nodes,used_nodes):

    # Génération des couples de noeuds
    edges = []
    better_path = []
    better_time = utils.calculate_time(path,times)
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

            tmp_time = utils.calculate_time(tmp_path,times)
            # Si la durée du nouveau chemin est améliorée, on enregistre le chemin
            if tmp_time<better_time :
                # Sauvegarde du chemin en comme un meilleur chemin
                better_path = [node for node in tmp_path]
                better_time= tmp_time

    if not better_path:
        better_path = [node for node in path]

    # Ajout des noeuds les plus proches et insérables TO DO: voir si déplacer en dehors de la fonction
    better_path,inserted = insert_nearest_free_node(nodes,better_path, tmax, times,profits,used_nodes)
    return better_path

def three_opt(path, tmax, profits, times, nodes, used_nodes):
    edges = []
    better_path = []
    better_time = utils.calculate_time(path, times)

    for i in range(len(path) - 1):
        edges.append((path[i], path[i+1]))

    for edge1, edge2, edge3 in combinations(edges, 3):
        if path.index(edge1[0]) >= path.index(edge2[0]) or path.index(edge2[0]) >= path.index(edge3[0]):
            break
        else:
            tmp_path = [node for node in path]

            tmp_path[path.index(edge1[1])], tmp_path[path.index(edge2[0])] = tmp_path[path.index(edge2[0])], tmp_path[path.index(edge1[1])]
            tmp_path[path.index(edge2[1])], tmp_path[path.index(edge3[0])] = tmp_path[path.index(edge3[0])], tmp_path[path.index(edge2[1])]

            tmp_time = utils.calculate_time(tmp_path, times)

            if tmp_time < better_time:
                better_path = [node for node in tmp_path]
                better_time = tmp_time

    if better_path == []:
        better_path = [node for node in path]

    better_path, inserted = insert_nearest_free_node(nodes, better_path, tmax, times, profits, used_nodes)
    return better_path

def insert_nearest_free_node(coords, path, tmax, times, profits, used_nodes):
    """Recherche des points les plus proches du chemin"""
    better_path = [node for node in path]
    node_inserted = True
    nearest_profit = 0
    while node_inserted:
        node_inserted = False
        for i in range(len(better_path) - 1):
            for new_node in coords:
                if new_node not in used_nodes:
                    # Calcul de la pénalité (detour) que fait coûter l'ajout du noeud
                    detour = utils.distance(better_path[i], new_node) + utils.distance(new_node, better_path[i + 1]) - utils.distance(better_path[i], better_path[i + 1])
                    if utils.calculate_time(better_path, times) + detour <= tmax and nearest_profit < profits[new_node]:
                        # Si l'ajout du noeud ne fait pas dépasser la durée maximale du trajet, on peut l'ajouter
                        nearest_node = new_node
                        nearest_profit = profits[new_node]
                        idx_to_be_placed = better_path.index(better_path[i]) + 1
                        node_inserted = True

        if node_inserted :
            better_path.insert(idx_to_be_placed, nearest_node)

    return better_path,node_inserted




