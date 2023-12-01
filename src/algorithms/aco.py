import numpy as np
import sys
sys.path.append('../')
import src.ressource.utils as utils
import src.algorithms.beasley as bsl
import src.algorithms.localSearch as localS
import math

Path = list[list[tuple[tuple[int,int]]]]
Profit = int

def ant_colony_optimization(graph : utils.Graph,
                            starting_point : int,
                            ending_point : int,
                            n_ants : int = 20 ,
                            n_iterations : int = 20,
                            Ncycles : int = 20,
                            alpha : float = 1,
                            beta : float = 30,
                            gamma : float = 0.05,
                            evaporation_rate : float = 0.98,
                            Pbest: float =0.05) -> tuple[Path, Profit]:
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

    n_points = len(graph.nodes)
    no_visit = []                     # Nœuds inaccessibles ou appartenant déjà à une tournée

    # ======================= Optimisation calculatoire ======================#
    # On supprime les noeuds inatteignables
    for point in graph.nodes :
        if point not in [graph.nodes[starting_point],graph.nodes[ending_point]] and graph.times[(graph.nodes[starting_point],point)] + graph.times[(point,graph.nodes[ending_point])] > graph.maxTime :
            no_visit.append(graph.nodes.index(point))
    # ========================================================================#

    sib = []  # iteration best solution
    sgb = []  # global best solution
    Nni = 0
    pheromone = np.ones((n_points, n_points))
    count = {}
    for node in graph.nodes:
        count[graph.nodes.index(node)] = 0
    for iteration in range(n_iterations):
        std_sol = [] # Solution au format standard (liste de tuples)

        for ant in range(n_ants):
            # ======================= Construction d'une solution =============================#
            sol_ant = constructSolution(graph ,no_visit,starting_point, ending_point, pheromone ,alpha, gamma, beta)
            # ========================================================================#

            #======================= Recherche Locale ====================================#
            used_nodes = set()
            for path in sol_ant:
                for node in path:
                    used_nodes.add(node)


            for i in range(len(sol_ant)):
                sol_ant[i] = localS.two_opt([graph.nodes[i] for i in sol_ant[i]], graph.maxTime, graph.profits, graph.times, graph.nodes,[graph.nodes[i] for i in list(used_nodes)+no_visit])
                for node in sol_ant[i]:
                    used_nodes.add(graph.nodes.index(node))
            # ========================================================================#
            std_sol.append(sol_ant)

            val = F(graph, sol_ant)
            for path in sol_ant:
                for node in path:
                    count[graph.nodes.index(node)]+= val


        #======================= Recherche de la meilleure solution trouvée par les fourmis ================================#
        sib = []
        max_quality = 0
        for solution in std_sol:
            quality = F(graph,solution)
            if quality > max_quality :
                max_quality = quality
                sib = solution
        # ==================================================================================================================#

        # ======================= Mise à jour de la meilleure solution ================================#
        if F(graph,sib) > F(graph,sgb):
            sgb = sib
            Nni = 0
        else :
            Nni += 1

        # ==================================================================================================================#

        #======================= Mise à jour des phéromones ================================#
        pheromoneUpdate(graph, pheromone, sgb, sib, evaporation_rate, Pbest, Nni, Ncycles, iteration,count)
        #========================================================================#

    return sgb, sum(utils.calculate_profit(path,graph.profits) for path in sgb)


def constructSolution(graph,no_visit, starting_point, ending_point,pheromone ,alpha, gamma, beta):
    """Construction d'une tournée. Méthode séquentielle """
    n_points = len(graph.nodes)
    solution = []
    used_nodes = [node for node in no_visit]
    for _ in range(graph.nbVehicules):
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
                used_nodes.append(next_point)

        # On signale que les sommets du chemin produit ne sont plus disponible
        solution.append(path)

    return solution


def pheromoneUpdate(graph,pheromone,sgb,sib,evaporation_rate,Pbest,Nni,Ncycles,iteration,count):
    """ Procédure de mise à jour des phéromones """
    n_points = len(graph.nodes)

    # Valeur maximale des phéromones sur un noeud
    t_max = F(graph, sgb) / (1 - evaporation_rate)

    # Valeur minimale des phéromones sur un noeud
    t_min = (1 - (Pbest ** (1 / n_points))) / (((n_points / 2) - 1) * (Pbest ** (1 / n_points))* t_max)

    if Nni == Ncycles:
        # Renouvellement des phéromones tout les Ncycles
        pheromone.fill(t_max)
    else:

        if iteration % 5 == 0:
            sol = sgb
        else:
            sol = sib
        used_nodes = []
        for path in sol :
            used_nodes += utils.extract_inner_tuples(path)
        for i in range(n_points - 1):
            for j in range(n_points):
                pheromone[i][j] = pheromone[i][j] * evaporation_rate + count[j]
                if pheromone[i][j] < t_min:
                    pheromone[i][j] = t_min
                elif pheromone[i][j] > t_max:
                    pheromone[i][j] = t_max


def F(graph : utils.Graph,sol : list[Path]):
    """Fonction de qualité : permet d'évaluer la qualité d'un chemin"""
    return sum(utils.calculate_profit(path[1:-1],graph.profits) for path in sol)/(sum(graph.profits[node] for node in graph.nodes[1:-1]))