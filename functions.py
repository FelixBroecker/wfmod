#!/usr/bin/env python3
import numpy as np

from csf import SelectedCI as sCI


class Functions(sCI):
    """Callable functions in wfmod input."""

    def csf2det(
        self,
        input_wavefunction,
        n_electrons,
        output_wavefunction="",
        verbose=False,
    ):
        """Convert CSF wave function to determinant wave function with a CI guess for the determiants."""
        # TODO multiply prefactors to get correct CI coefficient per determinant
        csf_coefficients, csfs, CI_coefficients, wfpretext = (
            self.read_AMOLQC_csfs(
                f"{input_wavefunction}.wf", n_electrons, verbose=verbose
            )
        )

        if verbose:
            print(f"Number of csfs: {len(csfs)}.")
            print("Convert CSFs to determinants.")

        CI_coefficients, _, dets = self.get_transformation_matrix(
            csf_coefficients, csfs, CI_coefficients
        )

        CI_coefficients = np.diagonal(CI_coefficients)
        CI_coefficients = [1 if n == 0 else 0 for n in range(len(dets))]
        csf_coefficients = []

        if verbose:
            print(f"Number of determinants: {len(dets)}.")
            print("Write wave function.")

        if not output_wavefunction:
            output_wavefunction = f"{input_wavefunction}_out"

        self.write_AMOLQC(
            csf_coefficients,
            dets,
            CI_coefficients,
            pretext=wfpretext,
            file_name=f"{output_wavefunction}.wf",
            wftype="det",
        )

    def cut(
        self,
        n_electrons,
        split_at,
        input_wavefunction,
        wftype="csf",
        output_wavefunction="",
        verbose=True,
    ):
        """Cut wavefunction after split_at."""
        csf_coefficients, determiants, CI_coefficients, wfpretext = (
            self.read_AMOLQC_csfs(
                f"{input_wavefunction}.wf", n_electrons, wftype=wftype
            )
        )

        if wftype == "det":
            CI_coefficients = [
                1 if n == 0 else 0 for n in range(len(determiants))
            ]

        if verbose:
            print("Write wave function.")

        try:
            if not output_wavefunction:
                output_wavefunction = f"{input_wavefunction}_out"

            self.write_AMOLQC(
                csf_coefficients[:split_at],
                determiants[:split_at],
                CI_coefficients[:split_at],
                pretext=wfpretext,
                file_name=f"{output_wavefunction}.wf",
                wftype=wftype,
            )
        except IndexError as e:
            print(
                f"Error in function cut: {e}: Maybe the wrong wftype is "
                "specified in the input file?"
            )

    def sort_wf(
        self,
        input_wavefunction,
        n_electrons,
        criterion,
        output_wavefunction="",
        initial_determinant=[],
        wftype="csf",
        verbose=True,
    ):
        """Sort wave function by criterion."""

        if wftype == "det":
            # TODO sorting of determinant wave functions
            print(
                "Sorting of determinant wave functions is not implemented yet.\n"
                "Please use the csf wave function type."
            )
            return

        csf_coefficients, determiants, CI_coefficients, wfpretext = (
            self.read_AMOLQC_csfs(
                f"{input_wavefunction}.wf", n_electrons, wftype=wftype
            )
        )

        if criterion == "ci_coefficient":
            # sort by CI coefficient
            if verbose:
                print("Sort wave function by absolute CI coefficient.")
            csf_coefficients, determiants, CI_coefficients = (
                self.sort_lists_by_list(
                    [csf_coefficients, determiants, CI_coefficients],
                    CI_coefficients,
                    side=-1,
                    absol=True,
                )
            )
        elif criterion == "by_excitation":
            if not initial_determinant:
                initial_determinant = self.build_energy_lowest_detetminant(
                    n_electrons
                )
            # sort by level of excitation
            if verbose:
                print("Sort wave function by level of excitation.")
            csf_coefficients, determiants, CI_coefficients = (
                self.sort_order_of_csfs(
                    csf_coefficients,
                    determiants,
                    CI_coefficients,
                    "by_excitation",
                    initial_determinant,
                )
            )
        elif criterion == "random":
            # sort in random order but keep ground state at first position
            if verbose:
                print("Sort wave function by level of excitation.")
            csf_coefficients, determiants, CI_coefficients = (
                self.sort_order_of_csfs(
                    csf_coefficients,
                    determiants,
                    CI_coefficients,
                    "random",
                    initial_determinant,
                )
            )

        if not output_wavefunction:
            output_wavefunction = f"{input_wavefunction}_out"

        self.write_AMOLQC(
            csf_coefficients,
            determiants,
            CI_coefficients,
            pretext=wfpretext,
            file_name=f"{output_wavefunction}.wf",
            wftype=wftype,
        )

    def det2csf(
        self,
        n_electrons,
        quantum_number_s,
        quantum_number_ms,
        input_wavefunction,
        output_wavefunction="",
        verbose=True,
    ):
        """Convert determinant wave funtion to CSF wave function by
        adding all missing determinants for the formation of CSFs."""
        # form csfs of these determinants
        _, determinants, _, wfpretext = self.read_AMOLQC_csfs(
            f"{input_wavefunction}.wf", n_electrons, wftype="det"
        )

        csf_coefficients, csfs = self.get_unique_csfs(
            determinants, quantum_number_s, quantum_number_ms
        )
        csf_coefficients, csfs = self.sort_determinants_in_csfs(
            csf_coefficients, csfs
        )
        CI_coefficients = [1 if n == 0 else 0 for n in range(len(csfs))]

        if verbose:
            print(
                f"number of csfs generated from {len(determinants)} determinants is "
                f"{len(csfs)}."
            )

        if not output_wavefunction:
            output_wavefunction = f"{input_wavefunction}_out"

        self.write_AMOLQC(
            csf_coefficients,
            csfs,
            CI_coefficients,
            pretext=wfpretext,
            file_name=f"{output_wavefunction}.wf",
            wftype="csf",
        )
