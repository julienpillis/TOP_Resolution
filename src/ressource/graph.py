
import random

class Graph :
    def __init__(self,nodes,times,profits,maxTime,nbVehicules):
        self.nodes = self.setNodes(nodes)
        self.times = times # faire la vérif
        self.profits = self.setProfits(profits)
        self.maxTime = maxTime
        self.nbVehicules = nbVehicules
        self.start_point = nodes[0]
        self.end_point = nodes[-1] # need to verify start/=end

    def getNodes(self): return self.nodes
    def getTimes(self): return self.times
    def getProfits(self): return self.profits
    def getMaxTime(self): return self.maxTime
    def getNbVehicules(self): return self.nbVehicules

    def setNodes(self,nodes):
        if len(set(nodes)) != len(nodes):
           raise Exception("Ensembles de noeuds différents")
        else :
            return nodes

    def setProfits(self, profits):
        if len(set(profits.keys())) != len(self.nodes):
            raise Exception("Ensembles de noeuds différents")
        else:
            return profits