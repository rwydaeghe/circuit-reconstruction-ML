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
        choices=["train", "evaluate"],
        help="operation to perform"
    )
    parser.add_argument(
        "method",
        choices=["regression", "gnn"],
        help="method to use"
    )
    parser.add_argument(
        "model",
        choices=["mlp", "linear", "lasso"],
        help="model to use",
        default = "mlp"
    )

    args = parser.parse_args()
    if args.operation == "train":

        if args.method == "gnn":
            input_graphs, target_graphs = circuitgen.data.get_gnn_data()
            circuitgen.gnn.train_gnn(input_graphs, target_graphs)
        else:
            training_method = getattr(circuitgen.train, args.method)
            training_model = getattr(circuitgen.models, args.model)
            input, output = circuitgen.data.get_regression_data()
            training_method(input, output, training_model,False)

    elif args.operation == "evaluate":
        if args.method == "gnn" or args.model != "mlp":
            print("cross validation only works for regression and self implemented models")

        else:
            input, output = circuitgen.data.get_regression_data()
            training_model = getattr(circuitgen.models, args.model)
            training_method = getattr(circuitgen.train, args.method)

            training_method(input, output, training_model,True)


if __name__ == "__main__":
    main()
