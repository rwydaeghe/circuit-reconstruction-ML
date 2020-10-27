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
        choices=["TODO"],
        help="model to use"
    )
    parser.add_argument(
        "source",
        help="source of input files to use"
    )
    parser.add_argument(
        "dest",
        help="dest of output netlist files"
    )
    # TODO add more args
    args = parser.parse_args()
    filename = "{}-{}".format(args.source, args.model)
    weights_filepath = os.path.join("weights", "{}.hdf5".format(filename))
    if args.operation == "train":
        circuitgen.train.train(weights_filepath)
    elif args.operation == "evaluate":
        circuitgen.train.evaluate_mlp(args.source)
    else:
        generated_data = circuitgen.generate.generate(weights_filepath)


if __name__ == "__main__":
    main()
