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
        coeffs_to_zero=True,
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
        if coeffs_to_zero:
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
                f"{input_wavefunction}.wf", n_electrons, wftype=wftype, split_at=split_at, verbose=verbose
            )
        )

        # if wftype == "det":
        #     CI_coefficients = [
        #         1 if n == 0 else 0 for n in range(len(determiants))
        #     ]

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

        # check if valid sort criterion
        valid_criteria = [
            "ci_coefficient",
            "by_excitation",
            "random",
            "unchanged",
        ]
        assert criterion in valid_criteria, f"Invalid sorting criterion: {criterion}"

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
        elif criterion == "unchanged":
            if verbose:
                print("Keep wave function order unchanged.")

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

    def add_singles(
        self,
        n_electrons,
        quantum_number_s,
        quantum_number_ms,
        n_orbitals,
        input_wavefunction,
        output_wavefunction="",
        frozen_electrons=[],
        frozen_MOs=[],
        orbital_symmetry=[],
        point_group="",
        wftype="csf",
        initial_determinant=[],
        wfpretext="",
        verbose=True,
    ):
        """Add all single excitations to the wavefunction"""
        csf_coefficients = []

        if not initial_determinant:
            # build energy lowest determinant
            initial_determinant = self.build_energy_lowest_detetminant(
                n_electrons
            )

        # build all single excitations with respect to
        # energy lowest determinant
        excited_determinants = self.get_excitations(
            n_orbitals,
            [1],
            initial_determinant,
            orbital_symmetry=orbital_symmetry,
            tot_sym=point_group,
            core=frozen_electrons,
            frozen_MOs=frozen_MOs,
        )

        # sort determinants in Amolqc format
        temp = []
        for det in excited_determinants:
            _, det_tmp = self.sort_determinant(1, det)
            temp.append(det_tmp)
        excited_determinants = temp.copy()
        print(
            f"Number of all singles in determinants: {len(excited_determinants)}"
        )

        if wftype == "csf":

            # get single csfs from determinant basis
            csf_coefficients_singles, csfs_singles = self.get_unique_csfs(
                excited_determinants.copy(),
                quantum_number_s,
                quantum_number_ms,
            )
            csf_coefficients_singles, csfs_singles = (
                self.sort_determinants_in_csfs(
                    csf_coefficients_singles, csfs_singles
                )
            )
            if verbose:
                print(f"Number of all singles in csfs: {len(csfs_singles)}")

            # read wave function
            csf_coefficients, csfs, CI_coefficients, wfpretext = (
                self.read_AMOLQC_csfs(
                    f"{input_wavefunction}.wf", n_electrons, wftype=wftype
                )
            )

            # remove singles from wave function
            degree_of_excitation = self.determine_excitations(
                csfs, initial_determinant, wf_type=wftype
            )
            csfs = [
                csf
                for csf, degree in zip(csfs, degree_of_excitation)
                if not degree == 1
            ]
            csf_coefficients = [
                coeff
                for coeff, degree in zip(
                    csf_coefficients, degree_of_excitation
                )
                if not degree == 1
            ]
            CI_coefficients = [
                coeff
                for coeff, degree in zip(CI_coefficients, degree_of_excitation)
                if not degree == 1
            ]
            CI_coefficients = (
                CI_coefficients[:1]
                + [0 for _ in excited_determinants]
                + CI_coefficients[1:]
            )

            # add singles to csf basis
            all_determinants = csfs[:1] + csfs_singles + csfs[1:]
            csf_coefficients = (
                csf_coefficients[:1]
                + csf_coefficients_singles
                + csf_coefficients[1:]
            )

        elif wftype == "det":
            # read wave function
            _, det_basis, CI_coefficients, wfpretext = self.read_AMOLQC_csfs(
                f"{input_wavefunction}.wf", n_electrons, wftype="det"
            )

            # remove singles from wave function
            degree_of_excitation = self.determine_excitations(
                det_basis, initial_determinant, wf_type=wftype
            )
            det_basis = [
                det
                for det, degree in zip(det_basis, degree_of_excitation)
                if not degree == 1
            ]
            CI_coefficients = [
                coeff
                for coeff, degree in zip(CI_coefficients, degree_of_excitation)
                if not degree == 1
            ]
            print(len(CI_coefficients), len(det_basis))

            # add singles to determinant basis
            all_determinants = (
                det_basis[:1] + excited_determinants + det_basis[1:]
            )

            CI_coefficients = (
                CI_coefficients[:1]
                + [0 for _ in excited_determinants]
                + CI_coefficients[1:]
            )

        # CI_coefficients = [
        #     1 if n == 0 else 0 for n in range(len(all_determinants))
        # ]

        if not output_wavefunction:
            output_wavefunction = f"{input_wavefunction}_add_sgls"

        self.write_AMOLQC(
            csf_coefficients,
            all_determinants,
            CI_coefficients,
            pretext=wfpretext,
            file_name=f"{output_wavefunction}.wf",
            wftype=wftype,
        )

    def wfFromConfs(
            self,
            n_electrons,
            quantum_number_s,
            quantum_number_ms,
            input_wavefunction,
            output_wavefunction="",
            wftype="csf",
            verbose=False):
        """Get configurations from wave function and take these to create a new wave function."""
        # get determinants from wave function

        if not output_wavefunction:
            output_wavefunction = f"{input_wavefunction}_from_confs"

        if wftype == "csf":
            self.csf2det(
                input_wavefunction,
                n_electrons,
                output_wavefunction=f"tmp",
                verbose=verbose,
                coeffs_to_zero=True,
            )
            _, determinants, CI_coefficients, wfpretext = (
                self.read_AMOLQC_csfs(
                    f"tmp.wf", n_electrons, wftype="det"
                )
                )
        else:
            _, determinants, CI_coefficients, wfpretext = (
                self.read_AMOLQC_csfs(
                    f"{input_wavefunction}.wf", n_electrons, wftype="det"
                ))

        # remove spin from determinants to get configurations
        configurations = self.get_configurations_from_determinants(determinants)

        # remove duplicate configurations
        configurations = self.remove_duplicate_configurations(configurations)

        # form new csfs based on this configurations
        csf_coefficients, csfs = self.get_unique_csfs(
            configurations, quantum_number_s, quantum_number_ms
        )

        # sort determinants in Amolqc format
        csf_coefficients, csfs = self.sort_determinants_in_csfs(
            csf_coefficients, csfs
        )

        if verbose:
            print()
            print(
                f"Number of csfs generated from {len(configurations)} configurations is "
                f"{len(csfs)}."
            )
            print("Write wave function.")
        # generate MO initial list
        CI_coefficients = [1 if n == 0 else 0 for n in range(len(csfs))]
        self.write_AMOLQC(
                csf_coefficients,
                csfs,
                CI_coefficients,
                pretext=wfpretext,
                file_name=f"{output_wavefunction}.wf",
            )
