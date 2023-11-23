from beasley import *
from utils import *
import time

if __name__ == "__main__":
    for c in list(map(chr, range(ord('b'), ord('r') + 1))):
        # Exemple d'utilisation
        file_path = f"Set_32_234/p1.4.{c}.txt"
        print(file_path)
        graph_object = read_file_and_create_graph(file_path)

        # Accéder aux données
        # print("Nodes:", graph_object.nodes)
        # print("Times:", graph_object.times)
        # print("Profits:", graph_object.profits)
        # print("Tmax", graph_object.getMaxTime())
        # print("NbVehicules:", graph_object.getNbVehicules())
        # print("NbNodes",  len(graph_object.nodes))

        # Exemple d'utilisation
        # points = [(0, 0), (1, 2), (2, 4), (3, 1), (4, 3)]
        # profits ={(0,0) : 0,(1,2) :0,(2,4) : 2, (3,1) : 3,(4,3) : 2}
        # graph = Graph(points,times_calculator(points),profits)

        # Temps de départ
        temps_debut = time.time()

        convoy, profit = beasley_top(graph_object, graph_object.getNodes()[0], graph_object.getNodes()[-1],
                                     graph_object.getMaxTime(), graph_object.getNbVehicules(), farthest_insertion)

        print(convoy, profit)
        # Temps final après execution
        temps_fin = time.time()

        # Durée d'execution
        temps_execution = temps_fin - temps_debut
        print("==============================")

        visualize_paths(graph_object.nodes, [path for path in convoy], profit, temps_execution)