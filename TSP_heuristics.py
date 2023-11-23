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
