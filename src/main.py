from algorithms.beasley import *
from ressource.utils import *
import time
from ressource.TSP_heuristics import *
import src.algorithms.aco as aco
from algorithms.gilletMiller import *
import src.algorithms.localSearch as localS

if __name__ == "__main__":
    for c in list(map(chr, range(ord('w'), ord('y') + 1))):
        # Exemple d'utilisation
        file_path = f"src/data/Set_66_234/p5.4.{c}.txt"
        print(file_path)
        graph_object = read_file_and_create_graph(file_path)

        print("Time MAX path :",graph_object.maxTime)
        # Temps de départ
        temps_debut = time.time()


        #convoy, profit = gillett_miller_top(graph_object, graph_object.getNodes()[0], graph_object.getNodes()[-1],
                                         #graph_object.getMaxTime(), graph_object.getNbVehicules(),localS.two_opt)
        #convoy, profit = gillett_miller_top_optimized(graph_object, graph_object.getNodes()[0], graph_object.getNodes()[-1],
                                            #graph_object.getMaxTime(), graph_object.getNbVehicules())
        convoy, profit = aco.ant_colony_optimization(graph_object,0,len(graph_object.nodes)-1)


        #print(profit)
        #convoy, profit = beasley_top(graph_object, graph_object.getNodes()[-1],graph_object.getNodes()[0],
                                     #graph_object.getMaxTime(), graph_object.getNbVehicules(), farthest_insertion,localS.three_opt)


        #print(profit)
        #convoy, profit = beasley_top_optimized(graph_object, graph_object.getNodes()[-1],graph_object.getNodes()[0],graph_object.getMaxTime(), graph_object.getNbVehicules())

        # Temps final après execution
        temps_fin = time.time()

        print(profit)
        print(convoy)



        # Durée d'execution
        temps_execution = temps_fin - temps_debut
        print("==============================")

        if convoy:
            #visualize_paths(graph_object.nodes, [path for path in convoy],convoy[0][0],convoy[0][-1],profit, temps_execution)
            drawGraph(graph_object,convoy,file_path,temps_execution,profit)