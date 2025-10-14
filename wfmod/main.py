#!/usr/bin/env python3

import sys
import yaml
from pyscript import *  # requirement pyscript as python package https://github.com/Leonard-Reuter/pyscript

from csf import SelectedCI
from automation import Automation
from evaluation import Evaluation
from wfmod.old.utils import Utils
from cipsi_jas import AddSingles
from functions import Functions

sCI = SelectedCI()
funcs = Functions()


def header():
    """Print header."""
    print()
    print(" " + "=" * 40)
    print(" Wave function generation and editation.")
    print(" " + "=" * 40)
    print()


def main():
    FUNCTIONS = {
        "generate_wavefunction": sCI.get_initial_wf,
        "csf2det": funcs.csf2det,
        "cut": funcs.cut,
        "sort_wf": funcs.sort_wf,
        "det2csf": funcs.det2csf,
        "add_singles": funcs.add_singles,
    }

    CLASS_REGISTRY = {
        "SelectedCI": SelectedCI,
        "Automation": Automation,
        "Evaluation": Evaluation,
        "Utils": Utils,
        "AddSingles": AddSingles,
    }

    # print header
    header()

    # if no input file is given, print usage and exit
    if len(sys.argv) == 1:
        sys.exit(
            """
        Script to generate wavefunctions for selected CI calculation and run CI calculations.

        usage: main.py <infile>

        with:
            <infile> being an .yaml file with all specification on molecule and demanded calculations.
    """
        )

    input_file = sys.argv[1]

    # read input file
    with open(input_file, "r") as reffile:
        input_data = yaml.safe_load(reffile)

    # call functions from input data
    cls = None
    for step in input_data.get("pipeline", []):
        func_name = step.get("function")
        class_name = step.get("class_name") or ""
        args = step.get("args", {})
        func = FUNCTIONS.get(func_name)

        if func_name == "initialize_class":
            if class_name in CLASS_REGISTRY:
                cls = CLASS_REGISTRY[class_name](**args)
                print(f"Class '{class_name}' has been initialized.")
                print()
            else:
                raise KeyError(f"Class {class_name} not found in registry.")

        elif class_name:

            assert class_name in CLASS_REGISTRY, "Class not found in registry."

            # check if class is initialized
            assert (
                cls is not None
            ), "Class is not initialized. Please initialize the class before calling methods."

            # call function from class
            try:
                print(f"Calling function '{class_name}.{func_name}':")
                print("-" * 21 + "-" * (len(class_name) + len(func_name)))

                if hasattr(cls, func_name):
                    getattr(cls, func_name)(**args)
                else:
                    raise AssertionError(
                        f"Function '{func_name}' not found in {cls}"
                    )

                print()
            except Exception as e:
                print(f"Error in function {func_name}: {e}")
                sys.exit(1)

        elif func:
            try:
                print(f"Calling function '{func_name}':")
                print("-" * 20 + "-" * len(func_name))
                func(**args)
                print()
            except Exception as e:
                print(f"Error in function {func_name}: {e}")
                sys.exit(1)


# program starts
main()
