'''
Created on 21.09.2018

@author: falensaa
'''

import networkx

from nparser.graph.mst import MaximumSpanningTreeAlgorithm

class SquareListScores(object):
    def __init__(self, length):
        self.outputs = [ ]
        self.scores = [ ]
        
        for _ in range(length):
            scoresRow = [ 0.0 ] * length
            outputRow = [ None ] * length
            self.outputs.append(outputRow)
            self.scores.append(scoresRow)
        
    def addScore(self, hId, dId, score):
        self.scores[hId+1][dId+1] = score
        
    def getScore(self, hId, dId):
        return self.scores[hId+1][dId+1]
    
    def addOutput(self, hId, dId, output):
        self.outputs[hId+1][dId+1] = output
        
    def getOutput(self, hId, dId):
        return self.outputs[hId+1][dId+1]
    
class SquareListScoresWithDims(SquareListScores):
    def __init__(self, length):
        super().__init__(length)
        self.dims = [ ]
        
        for _ in range(length):
            dimRow = [ None ] * length
            self.dims.append(dimRow)
        
    def addDim(self, hId, dId, dim):
        self.dims[hId+1][dId+1] = dim
        
    def getDim(self, hId, dId):
        return self.dims[hId+1][dId+1]
    
class ChuLiuEdmonds(MaximumSpanningTreeAlgorithm):
    
    def __init__(self):
        pass
        
    def emptyScores(self, instance, dim = False):
        if dim:
            return SquareListScoresWithDims(len(instance.sentence) + 1)
        
        return SquareListScores(len(instance.sentence) + 1)
     
    def handlesNonProjectiveTrees(self):
        return True
    
    def findMST(self, weights):
        graph = self.__buildInitialGraph(weights)
        return self.__runCLE(graph, len(graph.nodes()))
            
    def findMSTHeads(self, scores):
        tree = self.findMST(scores.scores)
        return self.__mstToHeads(tree)
     
    def __mstToHeads(self, tree):
        heads = [ -1 ] + [ list(tree.in_edges(d))[0][0] for d in sorted(list(tree.nodes()))[1:] ]
        return heads
    
    def __runCLE(self, graph, nodeNr):
        #TODO: creates a new graph
        bestParents = self.__getBestParentSubgraph(graph)
        cycle = self.__findCycle(bestParents)
        if cycle == [ ]:
            return bestParents
        else:
            #TODO: creates a new graph
            backtrackIn, backtrackOut = self.__contractCycle(graph, cycle, nodeNr)
            contractedMST = self.__runCLE(graph, nodeNr+1)
            self.__resolveCycle(contractedMST, cycle, nodeNr, backtrackIn, backtrackOut)
            return contractedMST
        
    def __buildInitialGraph(self, weights):
        g = networkx.DiGraph()
        g.add_nodes_from(range(len(weights)))
        edges = [(i, j, weights[i][j]) for i in range(len(weights)) for j in range(1, len(weights)) if i != j and j != 0 ]
        g.add_weighted_edges_from(edges)
        return g
    
    def __getBestParentSubgraph(self, graph):
        result = networkx.DiGraph()
        for node, attr in graph.nodes(data = True):
            if "in_cycle" in attr:
                continue
            
            heads = [ (attr["weight"], nodeH) for ( nodeH, _, attr) in graph.in_edges(node, data=True) if "in_cycle" not in attr ]
            
            if heads == [ ]:
                continue
            
            maxWeight, maxHead = max(heads)
            result.add_weighted_edges_from([ ( maxHead, node, maxWeight) ])
            
        return result

    def __findCycle(self, graph):
        cycleIter = networkx.simple_cycles(graph)
        
        try:
            firstCycle = next(cycleIter)
        except StopIteration:
            return [ ]
        
        # converts nodes to edges
        cycle = [ ]
        for i, nodeC in enumerate(firstCycle[:-1]):
            cycle.append((nodeC, firstCycle[i+1]))
        cycle.append((firstCycle[-1], firstCycle[0]))
        return cycle
    
    def __contractCycle(self, graph, cycle, cycleName):
        cycleNodes = [ e[0] for e in cycle ]
        cycleWeight = sum( graph[headC][nodeC]["weight"] for (headC, nodeC) in cycle )
        
        for node in cycleNodes:
            graph.node[node]['in_cycle'] = cycleName
            for h, d in graph.edges(node):
                graph[h][d]["in_cycle"] = cycleName
        
        graph.add_node(cycleName)
        
        # iterates over nodes outside the cycle
        backtrackIn = { }
        backtrackOut = { }
        for nodeG, attr in graph.nodes(data = True):
            if nodeG == cycleName or 'in_cycle' in attr:
                continue
            
            # edges leaving the cycle
            if nodeG != 0:
                bestWeight, bestNode = max([ (graph[nodeC][nodeG]["weight"], nodeC) for nodeC in cycleNodes ])
                graph.add_weighted_edges_from([(cycleName, nodeG, bestWeight)])
                backtrackOut[nodeG] = bestNode
                
            # edges entering the cycle
            bestWeight, bestNode = max([ ( graph[nodeG][nodeC]["weight"] + cycleWeight - graph[headC][nodeC]["weight"], 
                                           nodeC) for (headC, nodeC) in cycle ])
            
            graph.add_weighted_edges_from([(nodeG, cycleName, bestWeight)])
            backtrackIn[nodeG] = bestNode
            
        return backtrackIn, backtrackOut
    
    def __resolveCycle(self, mstRes, cycle, cycleNode, backtrackIn, backtrackOut):
        assert len(mstRes.in_edges(cycleNode)) == 1
        
        headOfCycle = list(mstRes.in_edges(cycleNode))[0][0]
        
        for headC, nodeC in cycle:
            mstRes.add_node(nodeC)
            
            # adding the only arc entering cycle
            if nodeC == backtrackIn[headOfCycle]:
                mstRes.add_edges_from([(headOfCycle, nodeC)])
            # all arcs inside the cycle
            else:
                mstRes.add_edges_from([(headC, nodeC)])
            
        # arcs leaving cycle
        for (cycleNode, nodeG) in mstRes.out_edges(cycleNode):
            nodeC = backtrackOut[nodeG]
            mstRes.add_edges_from([(nodeC, nodeG)])
            
        mstRes.remove_node(cycleNode)
        
        #TODO: move this into debug
        assert networkx.is_tree(mstRes)
        
