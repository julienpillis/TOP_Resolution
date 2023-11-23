import utils

"""
 *
 *
 * Différentes heuristiques de résolution du TSP
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
    n = len(points)

    pts = [point for point in points]
    # Choix arbitraire de deux points comme points de départ
    tour = [starting_point, ending_point]

    pts.pop(pts.index(starting_point)) # suppression de points[0]
    pts.pop(pts.index(ending_point)) # suppression de points[1]


    while len(pts) > 0 :
        # Trouver le point le plus éloigné
        max_dist = -1
        farthest_point = (0, 0)
        pt_tour = (0,0)
        for i in range(len(pts)):
            for point_tour in tour:
                if utils.distance(pts[i], point_tour) > max_dist:
                    max_dist = utils.distance(pts[i], point_tour)
                    farthest_point = pts[i]
                    pt_tour = point_tour

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
