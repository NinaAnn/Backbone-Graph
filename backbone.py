import networkx as nx
import cmath
import random
import numpy as np

def findMinTree(Data,mode='Inner'):
    '''
    params: 
        Data:   list [{'source': , 'target':}]
        mode:   'Inner': use minimum inner tree
                'Global': use minimum global tree
    return:
        G:  networkx.Graph()
        tree:   networkx.Graph()
        globalTreePaths:    2-dimentional binary ndarray, record each edge's tree path.
        edgeDict:   dictionary:{key=[source,target], value=int}, record the order of each edge in globalTreePaths
    '''
    G = readData(Data)
    tree = genMinInnerTree(G, mode)
    globalTreePaths, edgeDict = optimizeTree(G, tree)
    return G, tree, globalTreePaths, edgeDict

def readData(Data):
    '''
    params:
        Data:   list [{'source': , 'target':}]
    return:
        network: networkx.Graph()
    '''
    network = nx.Graph()
    for element in Data:
        network.add_edge(element['source'], element['target'], weight=element['value'])
    return network

def findTreeDis(s,t,tree):
    '''
    find minimum distance from s to t in tree, using dijkstra algorithm
    params:
        s:  string, source
        t:  string, target
        tree:   networkx.Graph()
    return:
        dis:    int
    '''
    path = nx.dijkstra_path(tree, source=s, target=t)
    dis = len(path)-1
    return dis

def genMinInnerTree(G, mode='Inner'):
    '''
    params:
        G:  networkx.Graph()
    return:
        tree:   networkx.Graph()
    '''
    tree = nx.Graph()
    graphNodes = list(G.nodes)
    firstNode = graphNodes[random.randint(0,len(graphNodes)-1)]
    tree.add_node(firstNode)
    TotalScale = len(list(G.nodes))
    while len(list(tree.nodes)) < TotalScale:
        maxName = findNextMaxDegree(tree, G)
        tree.add_node(maxName)
        findMinInnerEdge(tree, maxName, G, mode)
    print('-----------------------------')
    print('minimum inner tree generated')

    # print("{} {}".format(maxName, maxDegree))
    return tree

def genMinGlobalTree(G):

    tree = nx.Graph()
    graphNodes = list(G.nodes)
    firstNode = graphNodes[random.randint(0,len(graphNodes)-1)]
    tree.add_node(firstNode)
    TotalScale = len(list(G.nodes))
    while len(list(tree.nodes)) < TotalScale:
        maxName = findNextMaxDegree(tree, G)
        tree.add_node(maxName)

    return tree

def findNextMaxDegree(tree, G):
    '''
    params:
        tree, G
    return:
        maxName: string, name of the next node
    '''
    maxDegree = 0
    maxName = None
    for _, (name, degree) in enumerate(G.degree()):
        if name not in list(tree.nodes) and isTreeNeighbor(tree, name, G):
            if maxDegree < degree:
                maxDegree = degree
                maxName = name
    return maxName

def isTreeNeighbor(tree, node, G):
    '''
    decide weather node is in tree's neighborhood
    params:
        tree, G
        node: string, node name
    return:
        boolean
    '''
    treeNodes = list(tree.nodes)
    neighbors = G.neighbors(node)
    for element in neighbors:
        if element in treeNodes:
            return True
    return False

def calTreeQuality(tree, node=None, target=None, neighbors=None, G=None):
    '''
    calculate tree quality using initial formula
    params:
        tree:   networkx.Graph()
        node:   string, next node to be added
        target: string, the node in tree connected with 'node'
        neighbors:  list, neighbors of 'node' whose quality should be calculated
        G: networkx.Graph()
    return:
        quality: int
    '''
    newTree = tree.copy()
    newTree.add_edge(node, target, weight=G.edges[node, target]['weight'])
    quality = 0
    for neighbor in neighbors:
        quality += findTreeDis(neighbor, node, newTree)
    return quality

def calTreeQualityNew(tree, node=None, target=None, neighbors=None, G=None, mode='Inner'):
    '''
    calculate tree quality using new formula pow(dis,2)*weight
    params:
        tree:   networkx.Graph()
        node:   string, next node to be added
        target: string, the node in tree connected with 'node'
        neighbors:  list, neighbors of 'node' whose quality should be calculated
        G: networkx.Graph()
        mode:   string
    return:
        quality: int
    '''
    newTree = tree.copy()
    newTree.add_edge(node, target, weight=G.edges[node, target]['weight'])
    quality = 0
    if mode == 'Inner':
        for neighbor in neighbors:
            quality += pow(findTreeDis(neighbor, node, newTree),2) * G.edges[neighbor, node]['weight']

    return quality

def findMinInnerEdge(tree, nextNode=None, G=None, mode='Inner'):
    '''
    params:
        tree
        nextNode: string, next node name
        G
    return:
        None
    '''
    neighbor = G.neighbors(nextNode)
    treeNodes = list(tree.nodes)
    neighborInTree = []
    nextEdge = []
    minQuality = -1
    for _, node in enumerate(neighbor):
        if node in treeNodes:
            neighborInTree.append(node)
    for _, node in enumerate(neighborInTree):
        quality = calTreeQualityNew(tree, nextNode, node, neighborInTree, G, mode)
        if minQuality == -1:
            nextEdge = [nextNode, node]
            minQuality = quality
        elif minQuality < quality:
            nextEdge = [nextNode, node]
            minQuality = quality
    tree.add_edge(nextEdge[0], nextEdge[1], weight=G.edges[nextEdge[0], nextEdge[1]]['weight'])

def findMinGlobalEdge(tree, nextNode=None, G=None):
    neighbor = G.neighbors(nextNode)
    treeNodes = list(tree.nodes)
    neighborInTree = []
    for _,node in enumerate(neighbor):
        if node in treeNodes:
            neighborInTree.append(node)

def calTotalQuality(G, dic, globalPaths):
    '''
    params:
        G
        dic:    dictionary of all edges
        globalPaths:    2-dimentional ndarray 
    '''
    globalEdges = G.edges(data=True)
    quality = 0
    for (source, target, d) in globalEdges:
        dis = np.sum(globalPaths[dic[source, target]])
        quality += d['weight'] * pow(dis, 2)
    return quality
    
def optimizeTree(G, tree):
    '''
    params:
        G, tree
    return:
        globalTreePaths: 2-dimentional ndarray, containing all tree paths of all edges
        edgeDict:   dictionary:{key=[source,target], value=int}, record the order of each edge in globalTreePaths
    '''
    globalEdges = G.edges(data=True)
    dicLen = len(list(G.edges))
    edgeDict = {}
    treeEdges = tree.edges
    globalTreePaths = np.zeros((dicLen, dicLen), dtype=np.int)

    # Create global dictionary of total edges
    for i,(source, target, d) in enumerate(globalEdges):
        edgeDict[source, target] = i
    reEdgeDict = {v : k for k, v in edgeDict.items()}

    # Create global record of total edges' tree path in an ndarray
    for (source, target, d) in globalEdges:
        edge = (source, target)
        if edge not in treeEdges:
            path = nx.dijkstra_path(tree, source, target)
            pathDic = addPathIntoDic(edgeDict, path, dicLen)
            globalTreePaths[edgeDict[source, target]] = pathDic
        else:
            globalTreePaths[edgeDict[source, target]][edgeDict[source, target]] = 1

    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('optimization started')

    # Update 
    for (source, target, d) in globalEdges:
        edge = (source, target)
        if edge not in treeEdges:
            # reducedEdge = findReducedEdge(tree, edge, G, edgeDict, globalTreePaths)
            reducedEdge = findReducedEdgeNew(tree, edge, G, edgeDict, reEdgeDict, globalTreePaths)
            if reducedEdge != None:
                globalTreePaths = updatePath(globalTreePaths, edgeDict, reEdgeDict, reducedEdge, edge, tree, d, dicLen)
    print('optimization ended')
    return globalTreePaths, edgeDict

def updatePath(globalPaths, dic, redic, reducedEdge, addedEdge, tree, w, l):
    '''
    After selecting edge to be added and reduced, update tree and globalPaths
    params:
        globalPaths: 2-dimentional ndarray
        dic:    dictionary of edges {edge: int}
        redic:  reversed dictioanry of edges: {int: edge}
        reducedEdge:    (source, target)
        addedEdge:  (source, target)
        tree
        w:  {weight: int}
        l:  int, length of dictionary (number of edges)
    return:
        globalPaths
    '''
    tree.add_edge(addedEdge[0], addedEdge[1], weight = w['weight'])
    tree.remove_edge(reducedEdge[0],reducedEdge[1])
    globalPaths[dic[addedEdge]] = np.zeros((l,), dtype=np.int)
    globalPaths[dic[addedEdge]][dic[addedEdge]] = 1
    pathsConcerned = [num for num,value in enumerate(globalPaths) if value[dic[reducedEdge]] == 1]
    for element in pathsConcerned:
        edge = redic[element]
        path = nx.dijkstra_path(tree, edge[0], edge[1])
        pathDic = addPathIntoDic(dic, path, l)
        globalPaths[dic[edge[0], edge[1]]] = pathDic
    return globalPaths

def addPathIntoDic(dic, path, l):
    '''
    params:
        dic:    dictionary of all edges
        path:   list, [node1, node2, node3], all nodes passed from node1 to node3
        l:  int length of dic
    return:
        pathDic:    ndarray, binary map of the path
    '''
    pathDic = np.zeros((l,), dtype=np.int)
    for i,node in enumerate(path):
        if i < len(path)-1:
            if (node, path[i+1]) in dic:
                dicN = dic[node, path[i+1]]
            else:
                dicN = dic[path[i+1],node]
            pathDic[dicN] = 1
    return pathDic

def findReducedEdge(tree, edge, G, dic, globalPaths):
    '''
    params:
        tree, G, dic
        edge:   (source, target)
        globalPaths:    2-dimentional ndarray
    '''
    edgesTBReduced = nx.dijkstra_path(tree, edge[0], edge[1])
    edgePath = addPathIntoDic(dic, edgesTBReduced, len(list(G.edges)))
    edgeCircle = edgePath
    edgeCircle[dic[edge]] = 1
    minDeltaQ = 0
    reducedEdge = None
    for i,node in enumerate(edgesTBReduced):
        if i < len(edgesTBReduced)-1:
            newDeltaQ = 0
            edge = (node, edgesTBReduced[i+1])
            if edge in dic:
                edgeNum = dic[node, edgesTBReduced[i+1]]
            else:
                edgeNum = dic[edgesTBReduced[i+1], node]
                edge = (edgesTBReduced[i+1], node)

            pathsConcerned = [num for num,value in enumerate(globalPaths) if value[edgeNum] == 1]

            for element in pathsConcerned:
                UnionC = np.logical_and(globalPaths[element],edgePath)
                deltaDis = np.sum(edgeCircle) - 2*np.sum(UnionC)
                newDeltaQ += deltaDis
            
            if newDeltaQ < minDeltaQ:
                minDeltaQ = newDeltaQ
                reducedEdge = edge

    return reducedEdge

def findReducedEdgeNew(tree, edge, G, dic, redic, globalPaths):
    '''
    find reduced edge using new quality calculation method
    params:
        tree, G, dic
        edge:   (source, target)
        globalPaths:    2-dimentional ndarray
    '''
    edgesTBReduced = nx.dijkstra_path(tree, edge[0], edge[1])
    edgePath = addPathIntoDic(dic, edgesTBReduced, len(list(G.edges)))
    edgeCircle = edgePath
    edgeCircle[dic[edge]] = 1
    minDeltaQ = 0
    reducedEdge = None
    for i,node in enumerate(edgesTBReduced):
        if i < len(edgesTBReduced)-1:
            newDeltaQ = 0
            edge = (node, edgesTBReduced[i+1])
            if edge in dic:
                edgeNum = dic[node, edgesTBReduced[i+1]]
            else:
                edgeNum = dic[edgesTBReduced[i+1], node]
                edge = (edgesTBReduced[i+1], node)

            pathsConcerned = [num for num,value in enumerate(globalPaths) if value[edgeNum] == 1]

            for element in pathsConcerned:
                UnionC = np.logical_and(globalPaths[element],edgePath)
                deltaDis = np.sum(edgeCircle) - 2*np.sum(UnionC)
                oldDis = np.sum(globalPaths[element])
                eleWeight = G.edges[redic[element]]['weight']
                newDeltaQ += eleWeight*( pow((oldDis + deltaDis),2) - pow(oldDis,2))
            
            if newDeltaQ < minDeltaQ:
                minDeltaQ = newDeltaQ
                reducedEdge = edge

    return reducedEdge