from algorithms.beasley import *
from ressource.utils import *
import time
from ressource.TSP_heuristics import *
import src.algorithms.aco as aco
from algorithms.gillettMiller import *
import src.algorithms.localSearch as localS

if __name__ == "__main__":
    for c in list(map(chr, range(ord('d'), ord('d') + 1))):
        # Exemple d'utilisation
        file_path = f"src/data/Set_32_234/p1.2.{c}.txt"
        print(file_path)
        graph_object = read_file_and_create_graph(file_path)

        print("Time MAX path :",graph_object.maxTime)
        # Temps de départ
        temps_debut = time.time()


        #convoy, profit = gillett_miller_top(graph_object,localS.two_opt)
        #convoy, profit = gillett_miller_top_optimized(graph_object)
        #convoy, profit = beasley_top(graph_object, farthest_insertion,localS.three_opt)
        #convoy, profit = beasley_top_optimized(graph_object)
        convoy, profit = aco.ant_colony_optimization(graph_object,0,len(graph_object.nodes)-1)

        # Temps final après execution
        temps_fin = time.time()

        print("Profit = ",profit)
        print("Solution = ",convoy)



        # Durée d'execution
        temps_execution = temps_fin - temps_debut
        print("==============================")

        if convoy:
            drawGraph(graph_object,convoy,file_path,temps_execution,profit)