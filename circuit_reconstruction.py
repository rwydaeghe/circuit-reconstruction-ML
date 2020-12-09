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
        choices=["regression"],
        help="model to use"
    )
    parser.add_argument(
        "method",
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
    # TODO add more args
    args = parser.parse_args()
    # filename = "{}-{}".format(args.source, args.model)
    # weights_filepath = os.path.join("weights", "{}.hdf5".format(filename))
    if args.operation == "train":
        training_method = getattr(circuitgen.train, args.method)
        # training_model = getattr(circuitgen.models, args.method)
        input,output = circuitgen.data.get_regression_data()
        training_method(input, output)
    elif args.operation == "evaluate":
        input, output = circuitgen.data.get_regression_data()
        circuitgen.train.cross_Validation(input, output)



if __name__ == "__main__":
    main()
