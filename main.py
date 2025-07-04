#!/usr/bin/env python3

import sys
import numpy as np
import yaml
import json
from pyscript import *  # requirement pyscript as python package https://github.com/Leonard-Reuter/pyscript

from csf import SelectedCI
from automation import Automation
from evaluation import Evaluation
from utils import Utils
from cipsi_jas import AddSingles
from functions import Functions

evaluation = Evaluation()
utils = Utils()
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

    exit()

    # default parameter
    data = {
        "MoleculeInformation": {
            "numberOfElectrons": 0,
            "numberOfOrbitals": 0,
            "orbitalSymmetries": [],
            "pointGroup": "",
            "quantumNumber_S": 0,
            "quantumNumber_Ms": 0,
        },
        "WavefunctionOptions": {
            "wavefunctionName": "sCI",
            "wavefunctionOperation": "initial",
            "sort": "excitations",
            "initialDeterminant": [],
            "excitations": [],
            "frozenElectrons": [],
            "frozenMOs": [],
            "splitAt": 0,
            "maxCsfs": 1500,
            "wfType": "csf",
        },
        "Output": {
            "plotCICoefficients": False,
            "plotly": False,
        },
        "Specifications": {
            "criterion": "",
            "threshold": 1.0,
            "thresholdType": "cut_at",
            "keepMin": 0,
            "blocksize": 0,
            "nExpand": 0,
            "initialAMI": "",
            "iterationAMI": "",
            "finalAMI": "",
            "energyAMI": "",
            "keepAllSingles": False,
        },
        "Hardware": {"partition": "p16", "nTasks": "144"},
    }

    # load input in data
    for key, value in input_data.items():
        for sub_key, sub_value in value.items():
            data[key][sub_key] = sub_value

    # print header
    print(" " + "=" * 40)
    print(" Wave function generation and editation.")
    print(" " + "=" * 40)
    print()

    N = data["MoleculeInformation"]["numberOfElectrons"]
    n_MO = data["MoleculeInformation"]["numberOfOrbitals"]
    S = data["MoleculeInformation"]["quantumNumber_S"]
    M_s = data["MoleculeInformation"]["quantumNumber_Ms"]
    point_group = data["MoleculeInformation"]["pointGroup"]
    orbital_symmetry = data["MoleculeInformation"]["orbitalSymmetries"]

    wavefunction_name = data["WavefunctionOptions"]["wavefunctionName"]
    initial_determinant = data["WavefunctionOptions"]["initialDeterminant"]
    excitations = data["WavefunctionOptions"]["excitations"]
    frozen_electrons = data["WavefunctionOptions"]["frozenElectrons"]
    frozen_MOs = data["WavefunctionOptions"]["frozenMOs"]
    split_at = data["WavefunctionOptions"]["splitAt"]
    sort = data["WavefunctionOptions"]["sort"]
    max_csfs = data["WavefunctionOptions"]["maxCsfs"]
    wftype = data["WavefunctionOptions"]["wfType"]

    criterion = data["Specifications"]["criterion"]
    threshold = float(data["Specifications"]["threshold"])
    threshold_type = data["Specifications"]["thresholdType"]
    n_min = data["Specifications"]["keepMin"]
    blocksize = data["Specifications"]["blocksize"]
    n_expand = data["Specifications"]["nExpand"]
    initial_ami = data["Specifications"]["initialAMI"]
    iteration_ami = data["Specifications"]["iterationAMI"]
    energy_ami = data["Specifications"]["energyAMI"]
    final_ami = data["Specifications"]["finalAMI"]
    keep_all_singles = data["Specifications"]["keepAllSingles"]

    partition = data["Hardware"]["partition"]
    n_tasks = data["Hardware"]["nTasks"]

    auto = Automation(
        wavefunction_name,
        N,
        S,
        M_s,
        n_MO,
        excitations,
        orbital_symmetry,
        point_group,
        frozen_electrons,
        frozen_MOs,
        partition,
        n_tasks,
        criterion,
        blocksize,
        n_expand,
        sort,
        True,
        n_min,
        threshold,
        threshold_type,
        keep_all_singles,
        max_csfs,
    )

    # call demanded routine
    if data["WavefunctionOptions"]["wavefunctionOperation"] == "initial":
        ...

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "blockwise":
        ...

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "csf2det":
        ...

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "cut":
        ...

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "sort":
        ...

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "iterative":
        ...

    elif (
        data["WavefunctionOptions"]["wavefunctionOperation"] == "determine_exc"
    ):
        ...

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "exc":
        ...

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "add_singles":
        aS = AddSingles()
        aS.add_singles(
            N,
            S,
            M_s,
            n_MO,
            initial_determinant,
            orbital_symmetry,
            point_group,
            frozen_electrons,
            frozen_MOs,
            wavefunction_name,
            wftype,
        )

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "read_cipsi":
        wf_name_praefix = wavefunction_name.split(".")[0]
        # parse determinants and print them in AMOLQC format
        ci_coefficients, determinants = utils.parse_cipsi_dets(
            wavefunction_name
        )
        sCI.write_AMOLQC(
            [],
            determinants[:split_at],
            ci_coefficients[:split_at],
            pretext="",
            file_name=f"{wf_name_praefix}_dets.wf",
            wftype="det",
        )
        print(len(determinants))

        # get csfs from determinant basis and print wavefunction.
        # create guess for CI coefficients
        csf_coefficients, csfs = sCI.get_unique_csfs(determinants, S, M_s)
        csf_coefficients, csfs = sCI.sort_determinants_in_csfs(
            csf_coefficients, csfs
        )
        ci_csf_coefficients = [1 if n == 0 else 0 for n in range(len(csfs))]

        sCI.write_AMOLQC(
            csf_coefficients[:split_at],
            csfs[:split_at],
            ci_csf_coefficients[:split_at],
            pretext="",
            file_name=f"{wf_name_praefix}_csfs.wf",
        )
        print(f"len csfs: {len(csfs)}")

        # expand again in determinants to see how may
        # determinants have been added
        _, _, determinant_basis_csfs = sCI.get_transformation_matrix(
            csf_coefficients, csfs, range(len(csf_coefficients))
        )
        print(len(determinant_basis_csfs))

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "eval":
        """Evaluate blockwise optimization."""

        try:
            print("Create directory eval.")
            mkdir("eval")
        except FileExistsError:
            FileExistsError("Directory eval already exists.")

        # evaluate energy course during blockwise optimization
        energy_course_data = evaluation.get_energy_course()
        with open("eval/energy_course.txt", "w") as reffile:
            for line in energy_course_data:
                reffile.write(f"{line}\n")

        # evaluate excitaions degree of CSFs per wavefunction 1f each block

        list_of_excitation_lists = []
        list_of_counter_lists = []
        list_of_wavefunction_names = []

        for file in files():
            if file.endswith(".wf"):
                # get wavefunction name
                wavefunction_name = file.split(".")[0]

                try:
                    degree_of_excitation, counter = (
                        evaluation.get_excitations_degree(
                            N,
                            wavefunction_name,
                            initial_determinant,
                            wftype,
                            max_degree=20,
                            print_file=True,
                            verbose=False,
                        )
                    )
                    list_of_excitation_lists.append(degree_of_excitation)
                    list_of_counter_lists.append(counter)
                    list_of_wavefunction_names.append(wavefunction_name)
                except Exception as e:
                    continue

        with open("eval/excitation.json", "w") as reffile:
            json.dump(
                {
                    "wavefunction_name": list_of_wavefunction_names,
                    "degree_of_excitation": list_of_excitation_lists,
                    "counter": list_of_counter_lists,
                },
                reffile,
            )

    if data["Output"]["plotCICoefficients"]:
        if data["Output"]["plotly"]:
            evaluation.plot_ci_coefficients_plotly(wavefunction_name, N, n_MO)
        else:
            evaluation.plot_ci_coefficients(wavefunction_name, N)


# program starts


main()
