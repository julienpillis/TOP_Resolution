import math
import src.ressource.utils as utils
import src.algorithms.localSearch as localS

def gillett_miller_top(graph, localSearch):
    nodes = list(graph.getNodes())
    nodes.pop(nodes.index(graph.start_point))
    nodes.pop(nodes.index(graph.end_point))

    # Tri par ordre croissant polaire
    nodes.sort(key=lambda point: polar_order(graph.start_point, point), reverse=True)

    best_profit = 0
    best_solution = []

    for start_index in range(len(nodes)):
        # On va trouver une solution en commençant à chaque fois par un noeud différent
        paths = []
        current_arr = shift_array(nodes,start_index)
        for i in range(len(current_arr)):
            path = [graph.start_point, graph.end_point]
            j = i
            prev_node_idx = 0
            continue_insertion = True

            while continue_insertion:
                path.insert(prev_node_idx + 1, current_arr[j])
                duration = utils.calculate_time(path, graph.times)

                if duration <= graph.maxTime:
                    paths.append([node for node in path])
                    prev_node_idx += 1
                    j += 1

                    if j >= len(current_arr):
                        continue_insertion = False
                else:
                    continue_insertion = False

        solution, profit = utils.gen_conv(graph.nbVehicules, paths, graph.profits, graph.nodes)

        if solution == []:
            break
        else:
            # Optimisation des solutions
            profit = 0
            solution = list(solution)

            for i in range(len(solution)):
                used_nodes = utils.extract_inner_tuples(solution)
                better_path = localSearch(solution[i], graph.maxTime, graph.profits, graph.times, graph.nodes, used_nodes)

                if better_path:
                    solution[i] = [node for node in better_path]

                profit += utils.calculate_profit(solution[i], graph.profits)

            if profit > best_profit :
                best_solution = solution
                best_profit = profit

    return best_solution, best_profit

def polar_order(reference, point):
    # Fonction de calcul de l'angle polaire
    return (math.atan2(point[1] - reference[1], point[0] - reference[0]) + 2.0 * math.pi) % (2.0 * math.pi)

def shift_array(arr, start_index):
    # Vérifier que l'indice de départ est valide
    if start_index < 0 or start_index >= len(arr):
        raise ValueError("Indice de départ invalide")

    # Décaler le tableau
    shifted_array = arr[start_index:] + arr[:start_index]

    return shifted_array

def gillett_miller_top_optimized(graph : utils.Graph):
    """Application de toutes les heuristiques du PVC et de recherche locale lors de l'heuristique de Beasley"""
    best_convoy = []
    best_profit = 0
    for opt in localS.optimization :
        convoy,profit = gillett_miller_top(graph,opt)
        if profit > best_profit :
            best_convoy = convoy
            best_profit = profit
        reversed_convoy,reversed_profit = gillett_miller_top(graph,opt)
        if reversed_profit > best_profit:
            best_convoy = [list(reversed(path)) for path in reversed_convoy]
            best_profit = reversed_profit
    return best_convoy,best_profit