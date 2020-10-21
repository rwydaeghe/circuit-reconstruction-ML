# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:49:07 2020

@author: rbnwy
"""
import numpy as np
import random
import netlist as ntl
from netsolve import net_solve
import matplotlib.pyplot as plt

def add_component(component_type, component_value, file_name, allow_dangling_bonds=False):
    amount_of_components={'R': 0, 'L': 0, 'C': 0, 'I': 0}
    nodes=[]
    neighbours={}
    def connect(port1,port2):
        try:    
            neighbours[str(port1)] += 1
        except:
            neighbours[str(port1)]  = 1
        try:
            neighbours[str(port2)] += 1
        except:
            neighbours[str(port2)]  = 1
    with open(file_name,"r") as f:
        for n,line in enumerate(f):
            if line[0] == '.':
                line_to_add_component=n
                break
            elif line[0] != '*':
                items=line.split(' ')
                label=items[0]
                if amount_of_components[label[0]] < int(label[1]):
                    amount_of_components[label[0]] = int(label[1])
                port1=int(items[1])
                port2=int(items[2])
                if port1 not in nodes:
                    nodes.append(port1)
                if port2 not in nodes:
                    nodes.append(port2)
                connect(port1,port2)

    foo={k:v for k,v in neighbours.items() if v==1}
    if len(foo)==0:
        has_dangling_node=False
    elif len(foo)==1:
        has_dangling_node=True
        (dangling_node,_), = foo.items()    
    else:
        print('More than one dangling bond!')
        return

    if has_dangling_node:
        port1=dangling_node
    else:
        port1=random.choice(nodes)
    if allow_dangling_bonds:
        dangling_node=max(nodes)+1
        nodes.append(dangling_node)
    port2=random.choice([x for x in nodes if x != port1])
    connect(port1,port2)

    if component_type in amount_of_components:
        label=component_type+str(amount_of_components[component_type]+1)
    else:
        label=component_type+'1'
    
    lines=open(file_name, 'r').readlines()
    lines.insert(line_to_add_component, ' '.join([label,str(port1),str(port2),str(component_value),'\n']))
    open(file_name, 'w').write(''.join(lines))
       
def add_random_component(**kwargs):
    
    component_space={'R': np.linspace(1,10e0,10),
                     'L': np.linspace(1e-3,10e-3,10)}#,
                     #'C': np.linspace(1e-6,10e-6,10)} 
                     #C does not work yet since these will often introduce 
                     #floating nodes as there's no DC path to ground
    component_type=random.choice(list(component_space.keys()))
    component_value=random.choice(component_space[component_type])
    return add_component(component_type, component_value, **kwargs)

def generate_network(amount_of_components,**kwargs):
    #if you never want dangling bonds change the default argument in add_component
    for i in range(amount_of_components-1):
        add_random_component(**kwargs)
    add_random_component(allow_dangling_bonds=False,**kwargs)    

plt.close('all')

generate_network(10,file_name='my generated network.net')
net = ntl.Network('my generated network.net')
net_solve(net)
net.plot()

#manually clear the net file if you want to start from the basic network
