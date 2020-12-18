#! /usr/bin/env

import argparse
import os
import inspect
from pathlib import Path
import random
import numpy
import circuitgen

def main():
    parser = argparse.ArgumentParser(description="Use deep neural networks to reconstruct circuits")
    parser.add_argument(
        "operation",
        choices=["train", "evaluate", "generate"],
        help="operation to perform"
    )
    parser.add_argument(
        "model",
        choices=["mlp"],
        help="model to use"
    )

    # parser.add_argument(
    #     "source",
    #     help="source of input files to use"
    # )
    # parser.add_argument(
    #     "dest",
    #     help="dest of output netlist files"
    # )
    args = parser.parse_args()
    # filename = "{}-{}".format(args.source, args.model)
    # weights_filepath = os.path.join("weights", "{}.hdf5".format(filename))
    if args.operation == "train":
        training_method = getattr(circuitgen.train, args.model)
        training_model = getattr(circuitgen.models, args.model)
        input_graphs, target_graphs = circuitgen.data.get_gnn_data()
        circuitgen.gnn.train_gnn(input_graphs, target_graphs)
       # training_method(input, output, training_model)
    elif args.operation == "evaluate":
        input, output = circuitgen.data.get_regression_data()
        training_model = getattr(circuitgen.models, args.model)
        circuitgen.train.cross_Validation(input, output, training_model)


if __name__ == "__main__":
    main()
