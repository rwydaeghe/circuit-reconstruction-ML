import os
import glob

# read netlist into 2d array to use as input
def read_netlist(dirpath):
    components = []
    i = 0
    for file in glob.glob(os.path.join(dirpath,"*")):
        print("Parsing file {}: ".format(i) + file)
        i += 1
        current_file = open(file, "r")
        for line in current_file.readline():
            if line[0] != "*" and line[0] != ".":
                components.append(line.split(" "))
    return components
