from beasley import *
from utils import *
import time
from TSP_heuristics import *

if __name__ == "__main__":
    for c in list(map(chr, range(ord('a'), ord('r') + 1))):
        # Exemple d'utilisation
        file_path = f"Set_100_234/p4.3.{c}.txt"
        print(file_path)
        graph_object = read_file_and_create_graph(file_path)

        print("Time MAX path :",graph_object.maxTime)
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

        if convoy:
            visualize_paths(graph_object.nodes, [path for path in convoy],convoy[0][0],convoy[0][-1],profit, temps_execution)