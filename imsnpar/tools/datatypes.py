'''
Created on 21.08.2017

@author: falensaa
'''

from collections import deque

class Stack(object):
    def __init__(self, l):
        self.__stack = l
        
    def front(self):
        return self.__stack[0]
    
    def pop(self):
        return self.__stack.pop()
    
    def top(self):
        return self.__stack[-1]

    def tail(self):
        return self.__stack[:-1]
    
    def push(self, el):
        self.__stack.append(el)
        
    def length(self):
        return len(self.__stack)
        
    def toList(self):
        return self.__stack
        
    def empty(self):
        return len(self.__stack) == 0
    
    def elem(self, pos):
        return self.__stack[-(pos+1)]
    
    def __str__(self):
        return str(self.__stack)
    
    def __iter__(self):
        return self.__stack.__iter__()
    
class Buffer(object):
    def __init__(self, l):
        self.__buffer = deque(l)
        
    def empty(self):
        return len(self.__buffer) == 0
    
    def length(self):
        return len(self.__buffer)

    def head(self):
        return self.__buffer[0]
    
    def tail(self):
        return self.toList()[1:]
    
    def pop(self):
        return self.__buffer.popleft()

    def setFront(self, elem):
        self.__buffer[0] = elem
        
    def addFront(self, elem):
        self.__buffer.appendleft(elem)
        
    def append(self, elem):
        self.__buffer.append(elem)
        
    def toList(self):
        return list(self.__buffer)

    def elem(self, pos):
        return self.__buffer[pos]
    
    def __str__(self):
        return str(self.__buffer)
    
    def __iter__(self):
        return self.__buffer.__iter__()
    
class Tree(object):
    ROOT = -1
    NO_HEAD = -2
    
    def __init__(self, arcs, labels=None):
        self.__arcs = arcs
        self.__labels = labels
        
        # token -> nr of kids
        self.__nrOfChildren = self.__countChildren(arcs)
        
        # delayed -- maybe not needed
        self.__inOrder = None
        self.__isProjective = None
        
    def getChildren(self, headPos):
        result = [ ]
        for dPos, hPos in enumerate(self.__arcs):
            if headPos == hPos:
                result.append(dPos)
                
        return result
    
    def setLabels(self, labels):
        self.__labels = labels
        
    def getLabels(self):
        return self.__labels
    
    def areInOrder(self, headPos, depPos):
        if self.__inOrder == None:
            self.__inOrder = self.__traverseInOrder(self.__arcs)
            
        if headPos == self.ROOT:
            return True
        elif depPos == self.ROOT:
            return False
        else:
            return self.__inOrder[headPos] < self.__inOrder[depPos]
        
    def hasArc(self, headPos, depPos):
        if depPos == self.ROOT:
            return False
            
        return self.__arcs[depPos] == headPos

    def getLabel(self, depPos):
        if self.__labels == None or depPos == self.ROOT:
            return None
        
        return self.__labels[depPos]
    
    def getHead(self, depPos):
        if depPos == self.ROOT:
            return None
        
        return self.__arcs[depPos]
    
    def nrOfTokens(self):
        return len(self.__arcs) 
  
    def howManyChildren(self, headPos):
        return self.__nrOfChildren[headPos] if headPos in self.__nrOfChildren else 0
  
    def buildStrNoLabels(self):
        result = [ ]
        for (pos, head) in enumerate(self.__arcs):
            result.append("%s -> %s" % ( head, pos ))
            
        return str(result)
    
    def __countChildren(self, arcs):
        result = { }
        for head in arcs:
            if head not in result:
                result[head] = 1
            else:
                result[head] += 1
                
        return result
    
    def __str__(self):
        result = [ ]
        for (pos, head) in enumerate(self.__arcs):
            result.append("%s -> %s [%s]" % ( head, pos, str(self.getLabel(pos)) ))
            
        return str(result)
  
    def __iter__(self):
        return self.__arcs.values().__iter__()
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        
        if not self.nrOfTokens() == other.nrOfTokens():
            return False
             
        return all([ self.getHead(tok) == other.getHead(tok) and self.getLabel(tok) == other.getLabel(tok) for tok in range(self.nrOfTokens()) ])
   
    def toList(self):
        return self.__arcs
    
    def isProjective(self):
        if self.__isProjective is not None:
            return self.__isProjective
        
        # find transitive children
        transitiveChildren = self.buildTransitiveChildren()
        for head, children in transitiveChildren.items():
            for kid in children:
                span = range(kid+1, head) if kid < head else range(head+1, kid)
                for el in span:
                    if el not in children:
                        self.__isProjective = False
                        return False
                    
        self.__isProjective = True
        return True
    
    def buildTransitiveChildren(self):
        def updateTransitiveChildren(node, children):
            if node not in children:
                children[node] = set([])
                return
            
            nChildren = children[node].copy()
            for child in nChildren:
                updateTransitiveChildren(child, children)
                children[node].update(children[child])
            
        # start with leaves
        transitiveChildren = { }
        for (pos, head) in enumerate(self.__arcs): 
            if head not in transitiveChildren:
                transitiveChildren[head] = set()
                
            transitiveChildren[head].add(pos)
            
        # find transitive children
        updateTransitiveChildren(-1, transitiveChildren)
        return transitiveChildren
    
    def __traverseInOrder(self, arcs):
        def __doTraverseInorder(elem, allHeads, result):
            if elem not in allHeads:
                result.append(elem)
                return
            
            left, right = allHeads[elem]
            for leftChild in sorted(left):
                __doTraverseInorder(leftChild, allHeads, result)
            
            result.append(elem)
            
            for rightChild in sorted(right):
                __doTraverseInorder(rightChild, allHeads, result)
        
        children = { }
        for dep, head in enumerate(arcs):
            if head not in children:
                children[head] = ([ ], [ ])
                
            if dep < head:
                children[head][0].append(dep)
            else:
                children[head][1].append(dep)
            
        result = [ ]
        __doTraverseInorder(self.ROOT, children, result)
        
        assert result[0] == self.ROOT
        
        order = [ self.ROOT ] * self.nrOfTokens()
        for pos, head in enumerate(result[1:]):
            order[head] = pos
        
        return order
        
def sentence2Tree(sent):
    return Tree([ t.getHeadPos() for t in sent ], [ t.dep for t in sent ])