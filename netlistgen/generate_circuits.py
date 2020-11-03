# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:49:07 2020

@author: rbnwy
"""
import numpy as np
import random
from spicepy import netlist
from spicepy.netsolve import net_solve
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

class Read_and_write(object):
    def __init__(self,file_name,number_in_dataset=None):
        #reading
        self.file_name=file_name
        self.number_in_dataset=number_in_dataset
        self.amount_of_components = {'R': 0, 'L': 0, 'C': 0, 'I': 0}
        self.nodes = []
        self.neighbours = {}
        self.read_and_write_from_line = 0
        self.transient_data=None
        
        #writing
        self.allow_dangling_bonds=True
        """
        self.component_space = {'R': np.linspace(1, 10e0, 10),
                                'L': np.linspace(1e-3, 10e-3, 10)}  # ,
                              # 'C': np.linspace(1e-6,10e-6,10)}
        C does not work yet since these will often introduce 
        floating nodes as there's no DC path to ground
        """
        self.component_space = {'R': np.linspace(1, 10e0, 10),
                                'L': np.linspace(1e-3, 10e-3, 10),
                                'C': np.linspace(1e-6,10e-6,10)}
        
        
    def read_network(self):
        with open(self.file_name, "r") as f:
            for n, line in enumerate(f):
                if n>=self.read_and_write_from_line:
                    if line[0] == '.':
                        self.read_and_write_from_line = n
                        break
                    elif line[0] != '*':
                        items = line.split(' ')
                        label = items[0]
                        if self.amount_of_components[label[0]] < int(label[1:]):
                            self.amount_of_components[label[0]] = int(label[1:])
                        port1 = int(items[1])
                        port2 = int(items[2])
                        if port1 not in self.nodes:
                            self.nodes.append(port1)
                        if port2 not in self.nodes:
                            self.nodes.append(port2)
                        self.connect(port1, port2)
        
    def connect(self,port1, port2):
        try:
            self.neighbours[str(port1)] += 1
        except:
            self.neighbours[str(port1)] = 1
        try:
            self.neighbours[str(port2)] += 1
        except:
            self.neighbours[str(port2)] = 1
    
    def add_component(self,component_type, component_value):
        self.read_network()
        foo = {k: v for k, v in self.neighbours.items() if v == 1}
        if len(foo) == 0:
            has_dangling_node = False
        elif len(foo) == 1:
            has_dangling_node = True
            (dangling_node, _), = foo.items()
        else:
            print('More than one dangling bond!')
            return
    
        if has_dangling_node:
            port1 = dangling_node
        else:
            port1 = random.choice(self.nodes)
        port2_choices=[x for x in self.nodes if x != int(port1)]
        if self.allow_dangling_bonds:
            dangling_node = max(self.nodes) + 1
            port2_choices.append(dangling_node)
        port2 = random.choice(port2_choices)
        #self.connect(port1, port2)
    
        if component_type in self.amount_of_components:
            label = component_type + str(self.amount_of_components[component_type] + 1)
        else:
            label = component_type + '1'
        #self.amount_of_components[component_type]+=1
    
        lines = open(self.file_name, 'r').readlines()
        lines.insert(self.read_and_write_from_line, ' '.join([label, str(port1), str(port2), str(component_value), '\n']))
        open(self.file_name, 'w').write(''.join(lines))
        
    def add_random_component(self):
        component_type = random.choice(list(self.component_space.keys()))
        component_value = random.choice(self.component_space[component_type])
        return self.add_component(component_type, component_value)

    def compute_transients(self):
        self.net=netlist.Network(self.file_name)
        net_solve(self.net)
        self.transient_data=self.net.x
        
    def plot(self, content_list=None):
        if content_list is None:
            self.net.plot() #plots whatever is already in the netlist
        else:
            plot_command='.plot' + ' '.join(content_list) + '\n'
            lines = open(self.file_name, 'r').readlines()
            lines.append(plot_command)
            self.compute_transients() #I could probably not do this and use plot command
            self.net.plot()
        
    
    def write_more_network(self,amount_of_components):
        for i in range(amount_of_components - 1):
            self.add_random_component()
        if self.allow_dangling_bonds:
            self.allow_dangling_bonds=False
            self.add_random_component()
            self.allow_dangling_bonds=True
        else:
            self.add_random_component()
            
    def write_new_values(self):
        lines=open(self.file_name, "r").readlines()
        new_lines=[]
        for line in lines:
            new_line=line
            if line[0] != '*' and line[0] != '.':
                items = line.split(' ')
                component_type = items[0][0]
                if component_type != 'I' and component_type!='V':
                    new_value=random.choice(self.component_space[component_type])
                    new_line=line.replace(items[-1], str(new_value)+'\n')
            new_lines.append(new_line)
        open(self.file_name,'w').write(''.join(new_lines))
        
class Generate_data():
    def __init__(self, basic_file, target_dir='data/', seed=None):
        self.basic_file=basic_file
        self.target_dir=target_dir
        if target_dir[-1] != '/':
            self.target_dir+='/'
        self.read_and_write_objects=[]
        self.seed=seed
        random.seed(self.seed) #a None seed is actually no seed
                
    def clone_file(self, amount):
        copied_contents=open(self.basic_file,'r').readlines()
        for i in range(1,amount+1):
            name=self.target_dir+self.basic_file[:-4]+'_'+str(i)+'.net'
            self.read_and_write_objects.append(Read_and_write(name,i))
            open(name,'w').write(''.join(copied_contents))
            
    def delete_files(self):
        # please be careful with this command
        # e.g. if you set the target dir to a file containing code, you might lose it
        import os
        import glob
        
        for file in glob.glob(self.target_dir+'*'):
            #small protection
            if self.basic_file[:-4]+'_' in file:
                os.remove(file)
          
    def varied_values_data_set(self, size):
        self.clone_file(size)
        for rw_obj in tqdm(self.read_and_write_objects):
            rw_obj.write_new_values()
            rw_obj.compute_transients()
            transient_data_name=self.target_dir+self.basic_file[:-4]+'_trans_data_'+str(rw_obj.number_in_dataset)+'.pkl'
            with open(transient_data_name,'wb') as f:
                pickle.dump(rw_obj.transient_data,f)
                
    def varied_topology_data_set(self, size):
        print('To do!')
        #basically just use the write_more_network and clone_files methods
        #check for valid networks
    
plt.close('all')

data_set=Generate_data('RLC.net')
data_set.delete_files()
data_set.varied_values_data_set(100)
data_set.read_and_write_objects[0].plot()