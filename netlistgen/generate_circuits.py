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
    def __init__(self,file_name,number_in_dataset=None, allow_C=True):
        #reading
        self.file_name=file_name
        self.number_in_dataset=number_in_dataset
        self.amount_of_components = {'R': 0, 'L': 0, 'C': 0, 'V': 0, 'I': 0}
        self.nodes = []
        self.neighbours = {}
        self.read_and_write_from_line = 0
        self.transient_data=None
        self.is_singular=None
        
        #writing
        self.allow_dangling_bonds=True
        self.allow_C=allow_C
        if self.allow_C:
            self.component_space = {'R': np.linspace(1, 10e0, 10),
                                    'L': np.linspace(1e-3, 10e-3, 10),
                                    'C': np.linspace(1e-6,10e-6,10)}
        else:
            
            self.component_space = {'R': np.linspace(1, 10e0, 10),
                                    'L': np.linspace(1e-3, 10e-3, 10)}  # ,
                                  # 'C': np.linspace(1e-6,10e-6,10)}
            #C does not work yet since these will often introduce 
            #floating nodes as there's no DC path to ground
                    
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
    
        line=' '.join([label, str(port1), str(port2), str(component_value), '\n'])
        self.write_to_line(line, self.read_and_write_from_line)
        
    def add_random_component(self):
        component_type = random.choice(list(self.component_space.keys()))
        component_value = random.choice(self.component_space[component_type])
        return self.add_component(component_type, component_value)

    def compute_transients(self):
        self.net=netlist.Network(self.file_name)
        self.is_singular=False
        try:
            net_solve(self.net)
        except:
            print('Some exception occured trying to compute transients of file named '+self.file_name+'!')
            self.is_singular=True
        self.transient_data=self.net.x #TO DO: wrong, needs to be get_voltage
        
        # Catch singular matrices or other errors:
        if self.transient_data is not None:
            if np.nan in self.transient_data[:,0]: 
                print('Your matrix was probably singular as the results contain nan...')
                self.is_singular=True
        
    def plot(self, content_list=None):
        if content_list=='all_V':
            content_list=[]
            for component_type in self.amount_of_components.keys():
                for number in range(1,self.amount_of_components[component_type]+1):
                    content_list.append('v('+component_type+str(number)+')')

        if content_list=='all_I':
            content_list=[]
            for component_type in self.amount_of_components.keys():
                for number in range(1,self.amount_of_components[component_type]+1):
                    content_list.append('i('+component_type+str(number)+')')

        if content_list=='all':
            content_list=[]
            for component_type in self.amount_of_components.keys():
                for number in range(1,self.amount_of_components[component_type]+1):
                    content_list.append('v('+component_type+str(number)+')')
                    content_list.append('i('+component_type+str(number)+')')
                
        if content_list is None:
            self.net.plot() #plots whatever is already in the netlist
        else:
            plot_command='.plot ' + ' '.join(content_list) + '\n'
            self.write_to_line(plot_command, self.read_and_write_from_line+1)
            self.delete_last_line()
            self.compute_transients() #I could probably not do this and use plot command and self.transient_data
            # TO DO: figure out how to give current figure title
            try:
                self.net.plot()
                plt.title(self.file_name)
            except:
                pass
            if self.is_singular:
                plt.title('The matrix is singular which is why you do not see anything')
            plt.show()
            plt.pause(.1)
                
    def delete_last_line(self):
        lines = open(self.file_name, 'r').readlines()
        lines=lines[:-1]
        open(self.file_name, 'w').write(''.join(lines))
                
    def write_to_line(self, line, line_number):
        lines = open(self.file_name, 'r').readlines()
        lines.insert(line_number, line)
        open(self.file_name, 'w').write(''.join(lines))

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
    def __init__(self, basic_file, target_dir, seed=None):
        self.basic_file=basic_file
        self.target_dir=target_dir
        if target_dir[-1] != '/':
            self.target_dir+='/'
        self.read_and_write_objects=[]
        self.seed=seed
        random.seed(self.seed) #a None seed is actually no seed
        
    def read_dir(self):
        import glob
        for file in glob.glob(self.target_dir+'*.net'):
            if self.basic_file[:-len('.net')] in file:
                id_number=int(file.split(self.basic_file[:-len('.net')]+'_')[1].split('.net')[0])
                rw_obj=Read_and_write(file,id_number)
                self.read_and_write_objects.append(rw_obj)
                rw_obj.read_network()
                
    def clone_file(self, amount):
        copied_contents=open(self.basic_file,'r').readlines()
        for i in range(1,amount+1):
            name=self.target_dir+self.basic_file[:-len('.net')]+'_'+str(i)+'.net'
            self.read_and_write_objects.append(Read_and_write(name,i))
            open(name,'w').write(''.join(copied_contents))
            
    def delete_files(self):
        # please be careful with this command
        # deletes everything in the directory containing basic file name
        # e.g. if you set the target dir to a file containing code, you might lose it
        import os
        import glob
        
        for file in glob.glob(self.target_dir+'*'):
            #small protection
            if self.basic_file[:-len('.net')]+'_' in file:
                os.remove(file)
          
    def varied_values_data_set(self, size, write_to_pkl=True):
        self.clone_file(size)
        for rw_obj in tqdm(self.read_and_write_objects):
            rw_obj.write_new_values()
            rw_obj.compute_transients()
            transient_data_name=self.target_dir+self.basic_file[:-len('.net')]+'_trans_data_'+str(rw_obj.number_in_dataset)+'.pkl'
            if write_to_pkl:
                with open(transient_data_name,'wb') as f:
                    pickle.dump(rw_obj.transient_data,f)
                #to do: dump the time axis on here. Is it the same always?
        
    def varied_topology_data_set(self, size, allow_C):
        self.clone_file(size)
        for rw_obj in tqdm(self.read_and_write_objects):
            rw_obj.allow_C=allow_C
            rw_obj.write_more_network(int(random.random()*5))
            rw_obj.compute_transients()
            transient_data_name=self.target_dir+self.basic_file[:-len('.net')]+'_trans_data_'+str(rw_obj.number_in_dataset)+'.pkl'
            if write_to_pkl:
                with open(transient_data_name,'wb') as f:
                    pickle.dump(rw_obj.transient_data,f)
                #to do: dump the time axis on here. Is it the same always?
                
    def plot(self, data_files_ids, **kwargs):
        if data_files_ids=='all':
            data_files_ids=[]
            for rw_obj in self.read_and_write_objects:
                data_files_ids.append(rw_obj.number_in_dataset)
        for id_number in data_files_ids:
            for rw_obj in self.read_and_write_objects:
                if rw_obj.number_in_dataset == id_number:
                    rw_obj.plot(**kwargs)
                    break
                
    def collect_all_transients(self, v_or_i, arg):
        # this method assumes all data has same t vector, 
        # and the transients have already been computed,
        # and that arg is the same for all networks
        # so ideal for the varied_values_data_set 
        t=self.read_and_write_objects[0].net.t[1:]
        dt=t[1]-t[0]
        t-=dt/2
        data=np.zeros((len(t),len(self.read_and_write_objects)))
        for i,rw_obj in enumerate(self.read_and_write_objects):
            if v_or_i=='v':
                foo=rw_obj.net.get_voltage(arg)
                data[:,i]=foo[1:]
            elif v_or_i=='i':
                foo=rw_obj.net.get_current(arg)
                data[:,i]=foo[1:]
                
        return data, t
        
   
plt.close('all')

# CREATE

#you still have to create the target directory

#"""
data_set_1=Generate_data('RLC.net', target_dir='data/varied_values_data/', seed=1)
#data_set_1.delete_files() #be careful
data_set_1.varied_values_data_set(size=100)
#"""

"""
data_set_2=Generate_data('RLC.net', target_dir='data/varied_topology_data/', seed=2)
data_set_2.delete_files() #be careful
data_set_2.varied_topology_data_set(size=10, allow_C=False)
"""

# READ

#"""
#data_set_2=Generate_data('RLC.net', target_dir='data/varied_topology_data/', seed=2)
#data_set_1=Generate_data('RLC.net', target_dir='data/varied_values_data/', seed=1)
#data_set_2.read_dir()
#data_set_2.plot('all',content_list='all_I')
#"""

data_set_1.plot([1,2,3])
data, t = data_set_1.collect_all_transients('i','R1')
import scipy.io
scipy.io.savemat('varied_values_data_set.mat', {"data": data,"t": t})


