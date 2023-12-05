from itertools import combinations
import src.ressource.utils as utils

def two_opt(path,tmax, profits,times,nodes,used_nodes):
    """Recherche locale TWO-OPT avec insertion des noeuds atteignables"""

    edges = []
    better_path = []
    better_time = utils.calculate_time(path,times)
    optimized = True

    for i in range(len(path) - 1):
        # Génération des couples de noeuds
        edges.append((path[i], path[i+1]))

    while optimized :
        optimized = False
        for edge1,edge2 in combinations(edges,2) :
            # Si la première arête est après la deuxième ou si elles sont contigues, on ne respecte pas le critère
            if path.index(edge1[0])>=path.index(edge2[0]) or path.index(edge1[0])==path.index(edge2[0])+1  :
                break
            else :
                tmp_path = [node for node in path]

                # Croisement des arêtes
                tmp_path[tmp_path.index(edge1[1])],tmp_path[tmp_path.index(edge2[0])] = tmp_path[tmp_path.index(edge2[0])],tmp_path[tmp_path.index(edge1[1])]

                tmp_time = utils.calculate_time(tmp_path,times)
                # Si la durée du nouveau chemin est améliorée, on enregistre le chemin
                if tmp_time<better_time :
                    # Sauvegarde du chemin en comme un meilleur chemin
                    better_path = [node for node in tmp_path]
                    better_time= tmp_time
                    optimized = True

    if not better_path:
        better_path = [node for node in path]

    # Ajout des noeuds les plus proches et insérables TO DO: voir si déplacer en dehors de la fonction
    better_path,inserted = insert_nearest_free_node(nodes,better_path, tmax, times,profits,used_nodes)
    return better_path

def three_opt(path, tmax, profits, times, nodes, used_nodes):
    """Recherche locale THREE-OPT avec insertion des noeuds atteignables"""
    edges = []
    better_path = []
    better_time = utils.calculate_time(path, times)
    optimized = True

    for i in range(len(path) - 1):
        edges.append((path[i], path[i+1]))

    while optimized :
        optimized = False
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
                    optimized = True

    if better_path == []:
        better_path = [node for node in path]

    better_path, inserted = insert_nearest_free_node(nodes, better_path, tmax, times, profits, used_nodes)
    return better_path

def insert_nearest_free_node(nodes, path, tmax, times, profits, used_nodes):
    """Recherche des points les plus proches du chemin"""
    better_path = [node for node in path]
    node_inserted = True
    nearest_profit = 0
    while node_inserted:
        # On essaie d'ajouter des noeuds tant que l'on en a ajouté un à la boucle précédente
        node_inserted = False
        for i in range(len(better_path) - 1):
            for new_node in nodes:
                if new_node not in used_nodes:
                    # Calcul de la pénalité (detour) que fait coûter l'ajout du noeud
                    detour = utils.distance(better_path[i], new_node) + utils.distance(new_node, better_path[i + 1]) - utils.distance(better_path[i], better_path[i + 1])
                    if utils.calculate_time(better_path, times) + detour <= tmax and nearest_profit < profits[new_node]:
                        # Si l'ajout du noeud ne fait pas dépasser la durée maximale du trajet, on peut l'ajouter
                        nearest_node = new_node
                        nearest_profit = profits[new_node]
                        idx_to_be_placed = better_path.index(better_path[i]) + 1
                        node_inserted = True

        if node_inserted:
            if (nearest_node in better_path):
                better_path.remove(nearest_node)
            better_path.insert(idx_to_be_placed, nearest_node)

    return better_path,node_inserted

optimization = [two_opt,three_opt]