import os
import glob
import spicepy.netlist as ntl
from spicepy.netsolve import net_solve
import numpy as np


def read_netlist(dirpath):
    comp_to_id = {"R":1, "L": 2, "C": 3 , "V": 4}
    input = []
    output = []
    i = 0
    for file in glob.glob(os.path.join(dirpath, "*")):
        print("Parsing file {}: ".format(i) + file)
        i += 1
        current_file = open(file, "r")
        for line in current_file.readline():
            if line[0] != "*" and line[0] != ".":
                values = line.split(" ")
                values[0] = comp_to_id.get(values[0][0])
                input.append(values[:-1])
                output.append(values[-1])
    return input, output


def read_transiant_analyses(dirpath):
    # not sure how to read this data yet
    return

