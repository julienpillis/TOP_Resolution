import utils

"""
 *
 *
 * Différentes heuristiques constructives de résolution du TSP
 *
 *
"""

def nearest_neighbor(coords, start, end):
    """Heuristique du Plus Proche Voisin pour le TSP"""
    remaining_points = [coord for coord in coords]
    remaining_points.pop(remaining_points.index(start))
    remaining_points.pop(remaining_points.index(end))
    path = [start]

    while remaining_points:

        current_point = path[-1]
        nearest_point = min(remaining_points, key=lambda x: utils.distance(current_point, x))
        path.append(nearest_point)
        remaining_points.pop(remaining_points.index(nearest_point))

    path.append(end)
    return path

def farthest_insertion(points, starting_point, ending_point):
    """Heuristique d'insertion du plus point le plus lointain pour le TSP."""

    pts = [point for point in points]
    tour = [starting_point, ending_point]

    pts.pop(pts.index(starting_point)) # suppression de départ
    pts.pop(pts.index(ending_point)) # suppression de l'arrivée


    while len(pts) > 0 :
        # Trouver le point le plus éloigné
        max_dist = -1
        farthest_point = (0, 0)
        for i in range(len(pts)):
            for point_tour in tour:
                if utils.distance(pts[i], point_tour) > max_dist:
                    max_dist = utils.distance(pts[i], point_tour)
                    farthest_point = pts[i]

        # Trouver l'arête du tour actuel qui minimise l'ajout de longueur
        min_insertion_cost = float('inf')
        insert_index = 0
        for i in range(len(tour) - 1):
            # Variation de coût d'introduction du point le plus loin entre 2 noeuds
            d = utils.distance(tour[i], farthest_point) + utils.distance(farthest_point, tour[i + 1]) - utils.distance(tour[i],tour[i + 1])

            # Minimisation de la variation
            if d < min_insertion_cost:
                min_insertion_cost = d
                insert_index = i + 1

        # Insérer le point le plus éloigné à l'emplacement identifié
        tour.insert(insert_index, farthest_point)
        pts.pop(pts.index(farthest_point))

    return tour


def nearest_insertion(points, starting_point, ending_point):
    """Heuristique d'insertion du plus point le plus proche pour le TSP."""

    pts = [point for point in points]
    tour = [starting_point, ending_point]

    pts.pop(pts.index(starting_point)) # suppression du départ
    pts.pop(pts.index(ending_point)) # suppression de l'arrivée


    while len(pts) > 0 :
        # Trouver le point le plus proche
        min_dist = float("inf")
        nearest_point = (0, 0)
        for i in range(len(pts)):
            for point_tour in tour:
                if utils.distance(pts[i], point_tour) < min_dist:
                    min_dist = utils.distance(pts[i], point_tour)
                    nearest_point = pts[i]

        # Trouver l'arête du tour actuel qui minimise l'ajout de longueur
        min_insertion_cost = float('inf')
        insert_index = 0
        for i in range(len(tour) - 1):
            # Variation de coût d'introduction du point le plus loin entre 2 noeuds
            d = utils.distance(tour[i], nearest_point) + utils.distance(nearest_point, tour[i + 1]) - utils.distance(tour[i],tour[i + 1])

            # Minimisation de la variation
            if d < min_insertion_cost:
                min_insertion_cost = d
                insert_index = i + 1

        # Insérer le point le plus proche à l'emplacement identifié
        tour.insert(insert_index, nearest_point)
        pts.pop(pts.index(nearest_point))

    return tour


def best_insertion(points, starting_point, ending_point):
    """Heuristique de la meilleure insertion pour le TSP."""

    pts = [point for point in points]
    tour = [starting_point, ending_point]

    pts.pop(pts.index(starting_point)) # suppression du départ
    pts.pop(pts.index(ending_point)) # suppression de l'arrivée


    while len(pts) > 0 :

        # Trouver l'arête du tour actuel qui minimise l'ajout de longueur pour l'insertion du point
        min_insertion_cost = float('inf')
        insert_index = 0
        for i in range(len(tour) - 1):
            # Variation de coût d'introduction du point le plus loin entre 2 noeuds
            d = utils.distance(tour[i], pts[0]) + utils.distance(pts[0], tour[i + 1]) - utils.distance(tour[i],tour[i + 1])

            # Minimisation de la variation
            if d < min_insertion_cost:
                min_insertion_cost = d
                insert_index = i + 1

        # Insérer le point le plus proche à l'emplacement identifié
        tour.insert(insert_index, pts[0])
        pts.pop(0)

    return tour

def fletcher(points, starting_point, ending_point):
    pts = [point for point in points]

    pts.pop(pts.index(starting_point))  # suppression du départ
    pts.pop(pts.index(ending_point))  # suppression de l'arrivée

    times = utils.times_calculator(pts)
    edges = []
    times = sorted(times.items(), key=lambda t: t[1]) # Tri par ordre croissant des distances
    times = [time[0] for time in times]
    while len(times) > 0:
        used_nodes = extract_inner_tuples(edges)
        if times[0][0] not in used_nodes and times [0][1] not in used_nodes :
            # Si aucun noeud est extrémité d'une arête, on peut les ajouter sans problème
            edges.append((times[0][0],times[0][1]))
        elif times[0][0] in used_nodes and times [0][1] not in used_nodes:
            if used_nodes.count(times[0][0]) >= 2 :
                # Création d'une fourche si ajout d'un arc !
                pass
            else :
                edges.append((times[0][0], times[0][1]))
        elif times[0][1] in used_nodes and times [0][0] not in used_nodes:
            if used_nodes.count(times[0][1]) >= 2:
                # Création d'une fourche si ajout d'un arc !
                pass
            else:
                edges.append((times[0][0],times[0][1]))
        else :
            # Si les 2 noeuds sont déjà ajoutés
            if is_cycle(edges + [(times[0][0],times[0][1])]):
                # Si l'ajout de l'arête forme un cycle, on ne l'ajoute pas:
                pass
            elif used_nodes.count(times[0][0]) >= 2 or used_nodes.count(times[0][1]) >= 2:
                # Ajouter l'arc formerait une fourche !
                pass
            else :
                edges.append((times[0][0],times[0][1]))
        times.pop(0)


    # Réordonnancement du tableau
    no_match = [node for node in extract_inner_tuples(edges) if extract_inner_tuples(edges).count(node)==1] # Récupération des noeuds n'ayant pas 2 arêtes
    ordered_edges = [(starting_point,no_match[0])]
    while len(edges) > 0:
        match = [edge for edge in edges if edge[0]==ordered_edges[-1][1] or edge[1]==ordered_edges[-1][1]] # Récupération de l'arête ayant le même sommet que le 2ème noeud de la dernière arête
        if match[0][0]==ordered_edges[-1][1]:                                                              # S'il se trouve en première place du tuple, on l'ajoute
            ordered_edges.append(match[0])
        else :
            ordered_edges.append((match[0][1],match[0][0]))                                                 # Sinon, on inverse le tuple
        edges.remove(match[0])
    ordered_edges.append((ordered_edges[-1][1],ending_point))
    return [edge[0] for edge in ordered_edges]+[ending_point]



def extract_inner_tuples(array_of_tuples):
    """Extraire les tuples d'une liste de tuples"""
    inner_tuples = []
    for outer_tuple in array_of_tuples:
        for inner_tuple in outer_tuple:
            inner_tuples.append(inner_tuple)
    return inner_tuples

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            elif self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

def is_cycle(graph):
    coord_to_index = {}  # Dictionnaire pour mapper les coordonnées aux indices
    index = 0

    for edge in graph:
        for node in edge:
            if node not in coord_to_index:
                coord_to_index[node] = index
                index += 1

    uf = UnionFind(len(coord_to_index))

    for edge in graph:
        u, v = map(coord_to_index.get, edge)
        if uf.find(u) == uf.find(v):
            return True
        uf.union(u, v)

    return False