import numpy as np
import utils
import beasley as bsl
import math

Path = list[list[tuple[tuple[int,int]]]]
Profit = int

def F(graph : utils.Graph,path : Path):
    # TODO : modifier les indices(1,-1), vérifier la fonction
    return (utils.calculate_time(path[1:-1],graph.times))/sum(graph.profits[node] for node in graph.nodes[1:-1])

def ant_colony_optimization(graph : utils.Graph,
                            n_ants : int = 20 ,
                            n_iterations : int = 10,
                            alpha : float = 1,
                            beta : float = 0.5,
                            evaporation_rate : float = 0.98,
                            Q : float =0.5) -> tuple[Path, Profit]:
    """
    Meta-heuristique Ant colony optimization adaptée au problème du TOP

    :param graph : Le graph lié au problème
    :param n_ants: Le nombre de fourmis dans la colonie
    :param n_iterations : Le nombre d'itérations maximum
    :param alpha: Importante des phéromones (coefficient)
    :param beta : Importance de l'information heuristique (coefficient)
    :param evaporation_rate : Le taux d'évaporation
    :param Q : Le nombre de fourmis dans la colonie
    :returns : les meilleurs chemins et le profit total
    """


    points = graph.nodes
    n_points = len(points)
    best_paths = []
    used_nodes = []



    # ajouter la suppression des points inatteignables
    for route in range(graph.nbVehicules):
        sib = []  # iteration best solution
        sgb = []  # gloabl best solution
        Nni = 0
        pheromone = np.ones((n_points, n_points))
        best_path = []
        best_path_profits = 0
        for iteration in range(n_iterations):
            print(iteration)
            paths = []
            path_lengths = []

            for ant in range(n_ants):
                visited = [False] * n_points
                for i in used_nodes :
                    visited[i] = True
                current_point = 0
                # Initialisation du point de départ et d'arrivée
                visited[current_point] = True
                visited[-1] = True
                path = [current_point, len(points) - 1]
                path_length = 0

                while False in visited:
                    tmp_path = [i for i in path]
                    unvisited = np.where(np.logical_not(visited))[0]
                    probabilities = np.zeros(len(unvisited))

                    for i, unvisited_point in enumerate(unvisited):
                        """
                        probabilities[i] = (pheromone[current_point, unvisited_point] ** alpha) * (utils.distance(
                            points[current_point],
                            points[unvisited_point]) ** beta)  # à améliorer selon l'algo (le dividende)
                        """
                        # Calcul des probabilités
                        rj = graph.profits[graph.nodes[unvisited_point]]
                        cij = graph.times[(graph.nodes[current_point],graph.nodes[unvisited_point])]
                        cin = graph.times[(graph.nodes[current_point],graph.nodes[path[-1]])]
                        cjn = graph.times[(graph.nodes[unvisited_point], graph.nodes[path[-1]])]
                        teta_ij = (cij**2 + cin**2 -cjn**2)/(2*cij*cin)
                        neta = (rj/cij)*math.exp(teta_ij*0.5) # TODO: définir 0.5 en paramètre


                        probabilities[i] = (pheromone[current_point, unvisited_point] ** alpha) * (neta ** beta)
                    probabilities /= np.sum(probabilities)

                    next_point = np.random.choice(unvisited, p=probabilities)

                    tmp_path.insert(-1, next_point)
                    path_length += utils.distance(points[current_point], points[next_point])
                    visited[next_point] = True
                    current_point = next_point
                    if utils.calculate_time([graph.nodes[i] for i in tmp_path], graph.times) < graph.maxTime:
                        # On insère après le dernier sommet ajouté (pas en dernière position car il s'agit du sommet de destination)
                        path.insert(-1, next_point)

                path = bsl.two_opt([graph.nodes[i] for i in path], graph.maxTime, graph.profits, graph.times, graph.nodes,[graph.nodes[i] for i in path] + [graph.nodes[i] for i in used_nodes])
                path = [graph.nodes.index(node) for node in path]
                paths.append(path)
                path_lengths.append(path_length)


            sib = paths[np.argmax([F(graph,[graph.nodes[i] for i in path]) for path in paths])]
            if F(graph,[graph.nodes[i] for i in sib]) > F(graph,[graph.nodes[i] for i in sgb]):
                sgb = sib
                Nni = 0
            else :
                Nni += 1
                """
                if utils.calculate_profit([graph.nodes[i] for i in path], graph.profits) > best_path_profits:
                    best_path = path
                    best_path_profits = utils.calculate_profit([graph.nodes[i] for i in path], graph.profits)
                """
            #used_nodes = utils.extract_inner_tuples(paths)


            # Pheromone update

            t_max = F(graph,[graph.nodes[i] for i in sgb])/(1-evaporation_rate)
            Pbest = 0.5 # TODO: définir en paramètre
            t_min = (1-(Pbest**(1/n_points)))/((n_points/2 - 1)* (Pbest**(1/n_points)))*t_max

            if Nni==6 : # TODO: à modifier/vérifier, 250 par défaut
                for i in range(n_points) :
                    for j in range(n_points):
                        pheromone[i][j] = t_max
            else :
                for i in range(n_points-1) :
                    for j in range(n_points):
                        pheromone[i][j] = pheromone[i][j]*(1-evaporation_rate) + F(graph,[graph.nodes[i] for i in sgb]) if (i in sib and sib[sib.index(i) + 1]== j) else 0
                        if pheromone[i][j] < t_min :
                            pheromone[i][j] = t_min
                        elif pheromone[i][j] > t_max :
                            pheromone[i][j] = t_max
            """
            for path, path_length in zip(paths, path_lengths):
                for i in range(len(path) - 1):
                    pheromone[path[i], path[i + 1]] += Q / path_length
                pheromone[path[-1], path[0]] += Q / path_length
            """
        """
        for i in best_path:
            used_nodes.append(i)
        best_paths.append([graph.nodes[i] for i in best_path])
        """
        for i in sgb:
            used_nodes.append(i)
        best_paths.append([graph.nodes[i] for i in sgb])
    print(best_paths)

    #return [(best_paths[i], utils.calculate_profit(best_paths[i],graph.profits), utils.calculate_time(best_paths[i], graph.times)) for i in range(3)]
    return best_paths, sum([utils.calculate_profit(best_paths[i], graph.profits) for i in range(graph.nbVehicules)])
