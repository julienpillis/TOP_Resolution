from algorithms.beasley import *
from ressource.utils import *
import os
import time
import sys
import pandas as pd
from ressource.TSP_heuristics import *
import src.algorithms.aco as aco
from algorithms.gillettMiller import *
import src.algorithms.localSearch as localS

if __name__ == "__main__":
    ################ LOAD DATA  ####################
    # Get the current working directory
    data=[]
    current_path = os.getcwd()
    # Go up one directory to the parent directory
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    # Construct the path to the 'data' directory
    data_path = 'C:\\Users\\pllsj\\OneDrive\\Bureau\\UTC\\GI02\\AI09_RO06\\projet1\\src\\data'
    # Get a list of all items (files and folders) in the 'data' directory
    all_items = os.listdir(data_path)
    # Filter the items to include only folders that start with "Set"
    set_folders = [item for item in all_items if os.path.isdir(os.path.join(data_path, item)) and item.startswith('Set')]
    data_paths=[]
    for folder in set_folders:
        data_paths.append(os.path.join(data_path, folder))
    data=[]
    for directory_path in data_paths:
        #directory_path=data_paths[1]
        all_items = os.listdir(directory_path)
        txt_files = [os.path.join(directory_path, item) for item in all_items if os.path.isfile(os.path.join(directory_path, item)) and item.endswith('.txt')]
        ################ READ GRAPH ####################
        for file_path in txt_files:
            # Exemple d'utilisation
            #file_path = f"src/data/Set_66_234/p5.4.{c}.txt"
            print(file_path)
            graph_object = read_file_and_create_graph(file_path)
            print("Time MAX path :",graph_object.maxTime)
            # Temps de départ
            temps_debut = time.time()
            convoy, profit = aco.ant_colony_optimization(graph_object,0,len(graph_object.nodes)-1)
            temps_fin = time.time()
            if convoy:
                row={
                'dossier':directory_path.split('/')[-1],
                'fichier':file_path.split('/')[-1],
                'profit':profit,
                'temps d\'exécution':temps_debut-temps_fin,
                }
                data.append(row)
    csv_file_path = 'results/beasly_results.csv'
    df=pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)