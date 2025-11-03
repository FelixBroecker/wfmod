#!/usr/bin/env python3

import sys
import yaml
from pyscript import *  # requirement pyscript as python package https://github.com/Leonard-Reuter/pyscript

from csf import SelectedCI
from automation import Automation
from evaluation import Evaluation
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

def range_to_list(contracted_list: list) -> list:
    """Expand ranges in a list to full lists. E.g. ['1-2', 3, 5] -> [1, -1, 2, -2, 3, 5]"""
    for item in contracted_list:
        if isinstance(item, str) and "-" in item:
            start, end = map(int, item.split("-"))
            expanded = list(range(start, end + 1)) + list(range(-start, -end - 1, -1))
            contracted_list.remove(item)
            contracted_list.extend(expanded)
    contracted_list = sorted(contracted_list, key=abs)
    return contracted_list

def parse_input(input_data):
    """Parse input file and convert data if necessary."""
    for i, func in enumerate(input_data["pipeline"]):

        # check if short notation was used in frozen_MOs
        frozen = func.get("args", {}).get("frozen_MOs")
        if frozen and any(isinstance(x, str) for x in frozen):
            input_data["pipeline"][i]["args"]["frozen_MOs"] = range_to_list(frozen)

        # check if short notation was used in frozen_electrons
        frozen = func.get("args", {}).get("frozen_electrons")
        if frozen and any(isinstance(x, str) for x in frozen):
            input_data["pipeline"][i]['args']["frozen_electrons"] = range_to_list(frozen)

        # check if short notation was used in initial_determinant
        det = func.get("args", {}).get("initial_determinant")
        if det and any(isinstance(x, str) for x in det):
            input_data["pipeline"][i]['args']["initial_determinant"] = range_to_list(det)
        # check if short notation was used in excitations
        # must be performed after expanding frozen_MOs and frozen_electrons
        excitations = func.get("args", {}).get("excitations")
        if excitations and "fci" in excitations:
            n_frozen_elec = len(func["args"]["frozen_electrons"])
            n_frozen_orb = len(func["args"]["frozen_MOs"])
            n_elec = func["args"]["n_electrons"] - n_frozen_elec
            n_orb = func["args"]["n_orbitals"] - n_frozen_orb // 2
            max_exc = min(n_elec,  2 * n_orb - n_elec)
            input_data["pipeline"][i]['args']["excitations"] = [n for n in range(1, max_exc + 1)]

    return input_data


def main():
    FUNCTIONS = {
        "generate_wavefunction": sCI.get_initial_wf,
        "csf2det": funcs.csf2det,
        "cut": funcs.cut,
        "sort_wf": funcs.sort_wf,
        "det2csf": funcs.det2csf,
        "add_singles": funcs.add_singles,
        "wfFromConfs": funcs.wfFromConfs
    }

    CLASS_REGISTRY = {
        "SelectedCI": SelectedCI,
        "Automation": Automation,
        "Evaluation": Evaluation,
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
    input_data = parse_input(input_data)

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
