from graphviz import Digraph
from engine import Value


def trace(root):
        #Constructs node graph of all nodes and edges treating called object as root
        nodes,edges = set(),set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child,v))#edge between child and parent
                    build(child)
        
        build(root)
        nodegraph = (nodes,edges)
        return nodegraph

def draw_nodegraph(root):
    nodegraph = trace(root)
    dot = Digraph(format="svg",graph_attr={'rankdir':'LR'})

    (nodes,edges) = nodegraph
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid,label = "{%s | data %.4f | grad %.4f }" %(n.label, n.data, n.grad), shape = 'record')
        if n._op:
            #if the node was the result of an operation, record the operation
            dot.node(name=uid+n._op,label=n._op)
            dot.edge(uid + n._op,uid) #joins uid to uid op
    
    for n1,n2 in edges:
        #connect n1 to n2 via op node (n2 is the parent)
        dot.edge(str(id(n1)),str(id(n2))+n2._op) #connects n1 to n2 (parent) op node.
    
    dot.render(view=True)
    return dot