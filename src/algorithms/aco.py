import numpy as np
import sys
sys.path.append('../')
import src.ressource.utils as utils
import src.algorithms.beasley as bsl
import math

Path = list[list[tuple[tuple[int,int]]]]
Profit = int

def ant_colony_optimization(graph : utils.Graph,
                            starting_point : int,
                            ending_point : int,
                            n_ants : int = 20 ,
                            n_iterations : int = 100,
                            Ncycles : int = 2,
                            alpha : float = 1,
                            beta : float = 0.5,
                            gamma : float = 0.5,
                            evaporation_rate : float = 0.98,
                            Pbest: float =0.5) -> tuple[Path, Profit]:
    """
    Meta-heuristique Ant colony optimization adaptée au problème du TOP

    :param graph : Le graph lié au problème
    :param n_ants: Le nombre de fourmis dans la colonie
    :param n_iterations : Le nombre d'itérations maximum
    :param Ncycles : Seuil de réinitialisation des phéromones
    :param alpha: Importante des phéromones (coefficient)
    :param beta : Importance de l'information heuristique (coefficient)
    :param gamma : Influence de l'angle pour l'information heuristique (coefficient)
    :param evaporation_rate : Le taux d'évaporation
    :param Pbest : Probabilité de construire la meilleure solution trouvée quand les phéromones ont convergée vers Tmin ou Tmax
    :returns : les meilleurs chemins et le profit total
    """


    points = graph.nodes
    n_points = len(points)
    best_paths = []
    used_nodes = []                     # Nœuds inaccessibles ou appartenant déjà à une tournée

    # ======================= Optimisation calculatoire ======================#
    # On supprime les noeuds inatteignables
    for point in points :
        if point not in [graph.nodes[starting_point],graph.nodes[ending_point]] and graph.times[(graph.nodes[starting_point],point)] + graph.times[(point,graph.nodes[ending_point])] > graph.maxTime :
            used_nodes.append(graph.nodes.index(point))
    # ========================================================================#

    for route in range(graph.nbVehicules):
        sib = []  # iteration best solution
        sgb = []  # global best solution
        Nni = 0
        pheromone = np.ones((n_points, n_points))
        for iteration in range(n_iterations):
            paths = []
            norm_paths = []
            n_points = len(graph.nodes)
            for ant in range(n_ants):

                # ======================= Construct Solution =============================#
                path = constructSolution(graph ,starting_point, ending_point, used_nodes ,pheromone ,alpha, gamma, beta)
                # ========================================================================#

                #======================= Local Search ====================================#
                path = bsl.two_opt([graph.nodes[i] for i in path], graph.maxTime, graph.profits, graph.times, graph.nodes,[graph.nodes[i] for i in path] + [graph.nodes[i] for i in used_nodes])
                # ========================================================================#
                norm_paths.append(path)
                path = [graph.nodes.index(node) for node in path]
                paths.append(path)


            sib = paths[np.argmax([F(graph,[graph.nodes[i] for i in path]) for path in paths])]
            if F(graph,[graph.nodes[i] for i in sib]) > F(graph,[graph.nodes[i] for i in sgb]):
                sgb = sib
                Nni = 0
            else :
                Nni += 1

            #======================= Pheromone update ================================#
            pheromoneUpdate(graph, pheromone, sgb, sib, evaporation_rate, Pbest, Nni, Ncycles, iteration)
            # ========================================================================#

        for i in sgb:
            used_nodes.append(i)
        best_paths.append([graph.nodes[i] for i in sgb])
    return best_paths, sum([utils.calculate_profit(best_paths[i], graph.profits) for i in range(graph.nbVehicules)])


def constructSolution(graph, starting_point, ending_point, used_nodes ,pheromone ,alpha, gamma, beta):
    """Construction d'une tournée."""
    n_points = len(graph.nodes)

    # On signale
    visited = [False] * n_points
    for i in used_nodes:
        visited[i] = True
    current_point = starting_point
    visited[current_point] = True
    visited[ending_point] = True
    path = [current_point, ending_point]
    while False in visited:
        tmp_path = [i for i in path]
        unvisited = np.where(np.logical_not(visited))[0]
        probabilities = np.zeros(len(unvisited))

        for i, unvisited_point in enumerate(unvisited):
            # Calcul des probabilités de chaque noeud
            rj = graph.profits[graph.nodes[unvisited_point]]
            cij = graph.times[(graph.nodes[current_point], graph.nodes[unvisited_point])]
            cin = graph.times[(graph.nodes[current_point], graph.nodes[path[-1]])]
            cjn = graph.times[(graph.nodes[unvisited_point], graph.nodes[path[-1]])]
            teta_ij = (cij ** 2 + cin ** 2 - cjn ** 2) / (2 * cij * cin)
            neta = (rj / cij) * math.exp(teta_ij * gamma)
            probabilities[i] = (pheromone[current_point, unvisited_point] ** alpha) * (neta ** beta)
        probabilities /= np.sum(probabilities)

        # On détermine le prochain point
        next_point = np.random.choice(unvisited, p=probabilities)

        tmp_path.insert(-1, next_point)
        visited[next_point] = True
        current_point = next_point
        if utils.calculate_time([graph.nodes[i] for i in tmp_path], graph.times) < graph.maxTime:
            # On insère après le dernier sommet ajouté (pas en dernière position car il s'agit du sommet de destination)
            path.insert(-1, next_point)

    return path


def pheromoneUpdate(graph,pheromone,sgb,sib,evaporation_rate,Pbest,Nni,Ncycles,iteration):
    """ Procédure de mise à jour des phéromones """
    n_points = len(graph.nodes)

    # Valeur maximale des phéromones sur un noeud
    t_max = F(graph, [graph.nodes[i] for i in sgb]) / (1 - evaporation_rate)

    # Valeur minimale des phéromones sur un noeud
    t_min = (1 - (Pbest ** (1 / n_points))) / (((n_points / 2) - 1) * (Pbest ** (1 / n_points))) * t_max

    if Nni == Ncycles:
        # Renouvellement des phéromones tout les Ncycles
        pheromone.fill(t_max)
    else:
        if iteration % 5 == 0:
            path = sgb
        else:
            path = sib
        for i in range(n_points - 1):
            for j in range(n_points):
                pheromone[i][j] = pheromone[i][j] * (1 - evaporation_rate) + F(graph,[graph.nodes[i] for i in path]) if (i in path and path[path.index(i) + 1] == j) else 0
                if pheromone[i][j] < t_min:
                    pheromone[i][j] = t_min
                elif pheromone[i][j] > t_max:
                    pheromone[i][j] = t_max


def F(graph : utils.Graph,path : Path):
    """Fonction de qualité : permet d'évaluer la qualité d'un chemin"""
    return (utils.calculate_profit(path,graph.profits))/(sum(graph.profits[node] for node in graph.nodes))