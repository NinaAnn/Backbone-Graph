# Backbone-Graph
Realization of paper backbone extraction

Python: version 3

Only inner mode is realised (it seems like global mode has some problem)

## Usage
The usage of this function is:
```python
  findMinTree(Data,mode='Inner')
```
Input data type:
  list [{'source': , 'target':}]

Return:

  G: networkx.Graph()
  
  tree: networkx.Graph()
  
  globalTreePaths: two-dimentional binary ndarray, record each edge's tree path.
  
  edgeDict: dictionary:{key=[source,target], value=int}, record the order of each edge in globalTreePaths
  
