import time
import math
import numpy as np
from pyscript import *  # requirement pyscript as python package https://github.com/Leonard-Reuter/pyscript
from csf import SelectedCI


class Automation:
    def __init__(
        self,
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
        criterion: str,
        blocksize,
        n_expand,
        sort_option,
        verbose,
        n_min,
        threshold,
        threshold_type,
        keep_all_singles,
        max_csfs,
    ):
        self.sCI = SelectedCI()
        self.wavefunction_name = wavefunction_name
        self.N = N
        self.S = S
        self.M_s = M_s
        self.n_MO = n_MO
        self.excitations = excitations
        self.orbital_symmetry = orbital_symmetry
        self.point_group = point_group
        self.frozen_electrons = frozen_electrons
        self.frozen_MOs = frozen_MOs
        self.blocksize = blocksize
        self.n_expand = n_expand
        self.sort_option = sort_option
        self.verbose = verbose
        self.partition = partition
        self.n_tasks = n_tasks
        self.criterion = criterion
        self.n_min = n_min
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.keep_all_singles = keep_all_singles
        self.n_all_csfs = 0
        self.max_csfs = max_csfs

    def print_job_file(
        self,
        partition,
        job_name,
        n_tasks,
        ami_name,
        jobfile_name="amolqc_job",
        path="/home/broecker/bin/Amolqc/build/bin/amolqc",
    ):
        with open(f"{jobfile_name}", "w") as printfile:
            printfile.write(
                f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --output=o.%j
#SBATCH --ntasks={n_tasks}
#SBATCH --ntasks-per-core=1
# Befehle die ausgeführt werden sollen:
mpiexec -np {n_tasks} {path} {ami_name}.ami
"""
            )

    def check_job_done(self, amo_name, verbose=True):
        job_done = False
        try:
            with open(f"{amo_name}.amo", "r") as reffile:
                for line in reffile:
                    if "Amolqc run finished" in line:
                        job_done = True
                        if verbose:
                            print("job done.")
        except FileNotFoundError:
            pass
        return job_done

    def get_final_wavefunction(self, ami_name):
        last_wavefunction = ""
        wf_number = 0
        for file_name in ls():
            names = file_name.split(".")
            if names[-1] == "wf" and ami_name in names[0]:
                temp = names[0]
                number = int(temp.split("-")[-1])
                if number > wf_number:
                    wf_number = number
                    last_wavefunction = file_name
        return last_wavefunction.split(".")[0]

    def get_n_all_csfs(self, input_wf):
        """"""
        n_csfs = 0
        _, csfs, _, _ = self.sCI.read_AMOLQC_csfs(f"{input_wf}.wf", self.N)
        n_csfs += len(csfs)
        try:
            _, csfs_res, _, _ = self.sCI.read_AMOLQC_csfs(
                f"{input_wf}_res.wf", self.N
            )
            n_csfs += len(csfs_res)
        except FileNotFoundError:
            if self.verbose:
                print(f"{input_wf}_res.wf not found.")
        try:
            _, csfs_dis, _, _ = self.sCI.read_AMOLQC_csfs(
                f"{input_wf}_dis.wf", self.N
            )
            n_csfs += len(csfs_dis)
        except FileNotFoundError:
            print(f"{input_wf}_dis.wf not found.")
        return n_csfs

    def do_initial_block(self, block_label, initial_ami: str, energy_ami=""):
        dir_name = f"block{block_label}"
        #        mkdir(dir_name)

        with cd(dir_name):
            cp(f"../{self.wavefunction_name}.wf", ".")
            initial_determinant = self.sCI.build_energy_lowest_detetminant(
                self.N
            )
            self.sCI.get_initial_wf(
                self.S,
                self.M_s,
                self.n_MO,
                initial_determinant,
                self.excitations,
                self.orbital_symmetry,
                self.point_group,
                self.frozen_electrons,
                self.frozen_MOs,
                self.wavefunction_name,
                split_at=self.blocksize,
                sort_option=self.sort_option,
                verbose=self.verbose,
            )
            # extract total number of csfs
            rm(f"{self.wavefunction_name}.wf")
            mv(
                f"{self.wavefunction_name}_out.wf",
                f"{self.wavefunction_name}.wf",
            )
            cp(f"../{initial_ami}.ami", ".")
            # submit job
            self.print_job_file(
                self.partition,
                dir_name,
                self.n_tasks,
                initial_ami,
            )
            run("sbatch amolqc_job")
            # check if job done
            job_done = False
            while not job_done:
                job_done = self.check_job_done(initial_ami)
                if not job_done:
                    if self.verbose:
                        print("job not done yet.")
                    time.sleep(20)

            # get last wavefunction
            last_wavefunction = self.get_final_wavefunction(initial_ami)
            # compute energy criterion if required
            if self.criterion == "energy":
                mv(f"{self.wavefunction_name}.wf", "tmp")
                mv(
                    f"{last_wavefunction}.wf",
                    f"{self.wavefunction_name}.wf",
                )
                cp(f"../{energy_ami}.ami", ".")
                self.print_job_file(
                    self.partition,
                    f"e_{dir_name}",
                    self.n_tasks,
                    energy_ami,
                )
                run("sbatch amolqc_job")
                # check if job done
                job_done = False
                while not job_done:
                    job_done = self.check_job_done(energy_ami)
                    if not job_done:
                        if self.verbose:
                            print("job not done yet.")
                        time.sleep(20)
                mv(
                    f"{self.wavefunction_name}.wf",
                    f"{last_wavefunction}.wf",
                )
                mv("tmp", f"{self.wavefunction_name}.wf")
                cp(f"{energy_ami}.amo", f"../{dir_name}_nrg.amo")
                rm(f"{energy_ami}.ami")

            cp(f"{last_wavefunction}.wf", f"../{dir_name}.wf")
            try:
                cp(f"{self.wavefunction_name}_res.wf", f"../{dir_name}_res.wf")
            except FileNotFoundError:
                pass
            rm(f"{initial_ami}.ami")

        if self.verbose:
            print("finish initial block.")

    def do_block_iteration(
        self,
        n_blocks: int,
        input_wf: str,
        blockwise_ami: str,
        energy_ami="",
        ret=True,
    ):
        """"""
        if self.verbose:
            print(f"number of total blocks (without initial block) {n_blocks}")
        n_block = 0
        last_wavefunction = input_wf
        for i in range(n_blocks):
            if self.verbose:
                print()
                print(f"Block iteration: {i+1}/{n_blocks}")
                print()
            n_block += 1
            dir_name = f"block{n_block}"
            mkdir(dir_name)

            # get wavefunctions from previous iterations
            with cd(dir_name):
                try:
                    mv(f"../{last_wavefunction}_dis.wf", ".")
                except FileNotFoundError:
                    pass
                try:
                    mv(f"../{last_wavefunction}_nrg.amo", ".")
                except FileNotFoundError:
                    pass
                cp(f"../{last_wavefunction}.wf", ".")
                mv(f"../{last_wavefunction}_res.wf", ".")

                # get next block
                self.sCI.select_and_do_next_package(
                    self.N,
                    f"{last_wavefunction}_dis",
                    f"{last_wavefunction}",
                    f"{last_wavefunction}_res",
                    self.threshold,
                    self.criterion,
                    threshold_type=self.threshold_type,
                    split_at=self.blocksize,
                    n_min=self.n_min,
                    verbose=self.verbose,
                    n_expand=self.n_expand,
                )
                mv(
                    f"{last_wavefunction}_out.wf",
                    f"{self.wavefunction_name}.wf",
                )
                mv(
                    f"{last_wavefunction}_dis_out.wf",
                    f"{self.wavefunction_name}_dis.wf",
                )
                mv(
                    f"{last_wavefunction}_res_out.wf",
                    f"{self.wavefunction_name}_res.wf",
                )
                cp(f"../{blockwise_ami}.ami", ".")
                # submit job
                self.print_job_file(
                    self.partition,
                    dir_name,
                    self.n_tasks,
                    blockwise_ami,
                )
                run("sbatch amolqc_job")
                job_done = False
                while not job_done:
                    job_done = self.check_job_done(blockwise_ami)
                    if not job_done:
                        if self.verbose:
                            print("job not done yet.")
                        time.sleep(20)
                optimized_wavefunction = self.get_final_wavefunction(
                    blockwise_ami
                )

                # compute energy criterion if required
                if self.criterion == "energy":
                    mv(f"{self.wavefunction_name}.wf", "tmp")
                    mv(
                        f"{optimized_wavefunction}.wf",
                        f"{self.wavefunction_name}.wf",
                    )
                    cp(f"../{energy_ami}.ami", ".")
                    self.print_job_file(
                        self.partition,
                        f"e_{n_block}",
                        self.n_tasks,
                        energy_ami,
                    )
                    run("sbatch amolqc_job")
                    # check if job done
                    job_done = False
                    while not job_done:
                        job_done = self.check_job_done(energy_ami)
                        if not job_done:
                            if self.verbose:
                                print("job not done yet.")
                            time.sleep(20)
                    mv(
                        f"{self.wavefunction_name}.wf",
                        f"{optimized_wavefunction}.wf",
                    )
                    mv("tmp", f"{self.wavefunction_name}.wf")
                    cp(f"{energy_ami}.amo", f"../{dir_name}_nrg.amo")
                    rm(f"{energy_ami}.ami")

                # copy results to folder with all blocks
                cp(f"{optimized_wavefunction}.wf", f"../{dir_name}.wf")
                cp(
                    f"{self.wavefunction_name}_res.wf",
                    f"../{dir_name}_res.wf",
                )
                cp(
                    f"{self.wavefunction_name}_dis.wf",
                    f"../{dir_name}_dis.wf",
                )
                rm(f"{blockwise_ami}.ami")
                last_wavefunction = dir_name
                # check if there are still residual csfs or dets
                (
                    _,
                    csfs_residual,
                    _,
                    _,
                ) = self.sCI.read_AMOLQC_csfs(
                    f"{self.wavefunction_name}_res.wf", self.N
                )
                if len(csfs_residual) == 0:
                    print(
                        "Residual wavefunction is empty, thus the \
blockwise opimization is finished."
                    )
                    return dir_name
        if self.verbose:
            print("finish blockwise optimization")
        if ret:
            return dir_name

    def do_final_block(
        self,
        block_label: str,
        input_wf: str,
        final_ami: str,
    ):
        #
        # do finial selection from all CI coefficients
        #
        dir_name = f"block_{block_label}"
        mkdir(dir_name)
        with cd(dir_name):
            cp(f"../{input_wf}.wf", ".")
            mv(f"../{input_wf}_dis.wf", ".")
            try:
                mv(f"../{input_wf}_res.wf", ".")
            except FileNotFoundError:
                pass
            try:
                mv(f"../{input_wf}_nrg.amo", ".")
            except FileNotFoundError:
                pass
            #
            csf_coefficients, csfs, CI_coefficients, wfpretext = (
                self.sCI.read_AMOLQC_csfs(f"{input_wf}.wf", self.N)
            )
            csf_coefficients_dis, csfs_dis, CI_coefficients_dis, _ = (
                self.sCI.read_AMOLQC_csfs(f"{input_wf}_dis.wf", self.N)
            )

            energies = []
            energies_dis = []
            if self.criterion == "energy":
                _, energies_dis = self.sCI.parse_csf_energies(
                    f"{input_wf}_dis.wf",
                    len(csfs_dis),
                    sort_by_idx=True,
                    verbose=True,
                )
                _, energies = self.sCI.parse_csf_energies(
                    f"{input_wf}_nrg.amo",
                    len(csfs) - 1,
                    sort_by_idx=True,
                    verbose=True,
                )
                # add energy contribution for HF determinant, which shall
                # be largest contribution in the list. This excplicit
                # contribution is not physical but HF has largest
                # contribution to full wf.
                energies.insert(0, np.ceil(max(energies)))

            csf_coefficients += csf_coefficients_dis
            csfs += csfs_dis
            CI_coefficients += CI_coefficients_dis
            energies += energies_dis

            # keep all singe excitations
            idx = 0
            if self.keep_all_singles:
                initial_determinant = self.sCI.build_energy_lowest_detetminant(
                    self.N
                )
                csf_coefficients, csfs, CI_coefficients = (
                    self.sCI.sort_order_of_csfs(
                        csf_coefficients,
                        csfs,
                        CI_coefficients,
                        reference_determinant=initial_determinant,
                        option="by_excitation",
                    )
                )
                n_excitation = self.sCI.determine_excitations(
                    csfs, initial_determinant, "csf"
                )
                csf_coefficients, csfs, CI_coefficients, energies = (
                    self.sCI.sort_lists_by_list(
                        [csf_coefficients, csfs, CI_coefficients, energies],
                        n_excitation,
                    )
                )
                idx = 0
                for i, n_exc in enumerate(n_excitation):
                    if n_exc > 1:
                        idx = i
                        break

            ref_list = []
            absol = False
            if self.criterion == "energy":
                ref_list = energies
                absol = False
            elif self.criterion == "ci_coefficient":
                ref_list = CI_coefficients
                absol = True
                # sort csfs by criterion
            (
                csf_coefficients[idx:],
                csfs[idx:],
                CI_coefficients[idx:],
                energies[idx:],
            ) = self.sCI.sort_lists_by_list(
                [
                    csf_coefficients[idx:],
                    csfs[idx:],
                    CI_coefficients[idx:],
                    energies[idx:],
                ],
                ref_list,
                side=-1,
                absol=absol,
            )
            #
            self.sCI.write_AMOLQC(
                csf_coefficients[: self.blocksize],
                csfs[: self.blocksize],
                CI_coefficients[: self.blocksize],
                pretext=wfpretext,
                file_name=f"{self.wavefunction_name}.wf",
            )
            self.sCI.write_AMOLQC(
                csf_coefficients[self.blocksize :],
                csfs[self.blocksize :],
                CI_coefficients[self.blocksize :],
                energies=energies[self.blocksize :],
                file_name=f"{self.wavefunction_name}_dis.wf",
            )

            cp(f"../{final_ami}.ami", ".")
            # submit job
            self.print_job_file(
                self.partition,
                dir_name,
                self.n_tasks,
                final_ami,
            )
            run("sbatch amolqc_job")
            # check if job done
            job_done = False
            while not job_done:
                job_done = self.check_job_done(final_ami)
                if not job_done:
                    if self.verbose:
                        print("job not done yet.")
                    time.sleep(20)

            # get last wavefunction and copy to folder with all blocks
            last_wavefunction = self.get_final_wavefunction(final_ami)
            cp(f"{last_wavefunction}.wf", f"../{dir_name}.wf")
            rm(f"{final_ami}.ami")
            if self.verbose:
                print()
                print("finish final block.")

    def do_final_iteration(
        self,
        label: str,
        input_wf: str,
        max_csfs: int,
        final_ami: str,
        energy_ami="",
    ):
        #
        # add discarded csfs until max csfs number
        #
        dir_name = f"{label}"
        # mkdir(dir_name)
        with cd(dir_name):
            # cp(f"../{input_wf}.wf", ".")
            # mv(f"../{input_wf}_dis.wf", ".")
            # try:
            #    mv(f"../{input_wf}_nrg.amo", ".")
            # except FileNotFoundError:
            #    pass
            ##
            # csf_coefficients, csfs, CI_coefficients, wfpretext = (
            #    self.sCI.read_AMOLQC_csfs(f"{input_wf}.wf", self.N)
            # )
            # csf_coefficients_dis, csfs_dis, CI_coefficients_dis, _ = (
            #    self.sCI.read_AMOLQC_csfs(f"{input_wf}_dis.wf", self.N)
            # )

            ## add current csfs with already sorted, discarded csfs
            # csf_coefficients += csf_coefficients_dis
            # csfs += csfs_dis
            # CI_coefficients += CI_coefficients_dis

            # self.sCI.write_AMOLQC(
            #    csf_coefficients[:max_csfs],
            #    csfs[: max_csfs],
            #    CI_coefficients[: max_csfs],
            #    pretext=wfpretext,
            #    file_name=f"{self.wavefunction_name}.wf",
            # )
            # self.sCI.write_AMOLQC(
            #    csf_coefficients[max_csfs :],
            #    csfs[max_csfs :],
            #    CI_coefficients[max_csfs :],
            #    file_name=f"{self.wavefunction_name}_dis.wf",
            # )

            ## optimize wavefunction
            # cp(f"../{final_ami}.ami", ".")
            ## submit job
            # self.print_job_file(
            #    self.partition,
            #    dir_name,
            #    self.n_tasks,
            #    final_ami,
            # )
            # run("sbatch amolqc_job")
            ## check if job done
            # job_done = False
            # while not job_done:
            #    job_done = self.check_job_done(final_ami)
            #    if not job_done:
            #        if self.verbose:
            #            print("job not done yet.")
            #        time.sleep(20)

            # get last wavefunction and copy to folder with all blocks
            optimized_wavefunction = self.get_final_wavefunction(final_ami)

            # compute energy criterion if required
            if self.criterion == "energy":
                mv(f"{self.wavefunction_name}.wf", "tmp")
                mv(
                    f"{optimized_wavefunction}.wf",
                    f"{self.wavefunction_name}.wf",
                )
                cp(f"../{energy_ami}.ami", ".")
                self.print_job_file(
                    self.partition,
                    f"e_{label}",
                    self.n_tasks,
                    energy_ami,
                )
                run("sbatch amolqc_job")
                # check if job done
                job_done = False
                while not job_done:
                    job_done = self.check_job_done(energy_ami)
                    if not job_done:
                        if self.verbose:
                            print("job not done yet.")
                        time.sleep(20)
                mv(
                    f"{self.wavefunction_name}.wf",
                    f"{optimized_wavefunction}.wf",
                )
                mv("tmp", f"{self.wavefunction_name}.wf")
                rm(f"{energy_ami}.ami")
                # parse energy contributions
                _, energies_optimized = self.parse_csf_energies(
                    f"{energy_ami}.amo",
                    len(csfs[:max_csfs]) - 1,
                    sort_by_idx=True,
                    verbose=True,
                )

            # read in new optimized wavefunction
            (
                csf_coefficients_optimized,
                csfs_optimized,
                CI_coefficients_optimized,
                wfpretext,
            ) = self.sCI.read_AMOLQC_csfs(
                f"{optimized_wavefunction}.wf", self.N
            )

            # select by criterion
            ref_list_optimized = []
            mask = [True for _ in range(len(ref_list_optimized))]
            absol = False
            if criterion == "energy":
                # add energy contribution for HF determinant, which shall
                # be largest contribution in the list. This excplicit contribution
                # is not physical but HF has largest contribution to full wf.
                energies_optimized.insert(0, np.ceil(max(energies_optimized)))
                ref_list_optimized = energies_optimized
                absol = False
                mask = [False] + [
                    True for _ in range(len(ref_list_optimized) - 1)
                ]
            elif criterion == "ci_coefficient":
                ref_list_optimized = CI_coefficients_optimized
                mask = [True for _ in range(len(ref_list_optimized))]
                absol = True
            # cut of csfs by CI coefficient threshold
            tmp_first, tmp_scnd = self.cut_lists(
                [
                    csf_coefficients_optimized,
                    csfs_optimized,
                    CI_coefficients_optimized,
                    energies_optimized,
                ],
                ref_list_optimized,
                threshold,
                threshold_type=threshold_type,
                mask=mask,
                side=-1,
                absol=absol,
            )
            (
                csf_coefficients_selected,
                csfs_selected,
                CI_coefficients_selected,
                energies_selected,
            ) = tmp_first
            (
                csf_coefficients_discarded,
                csfs_discarded,
                CI_coefficients_discarded,
                energies_discarded,
            ) = tmp_scnd

            # write wavefunction
            self.write_AMOLQC(
                csf_coefficients_selected,
                csfs_selected,
                CI_coefficients_selected,
                pretext=wfpretext,
                file_name=f"{filename_optimized}_selec.wf",
            )
            self.write_AMOLQC(
                csf_coefficients_discarded,
                csfs_discarded,
                CI_coefficients_discarded,
                energies=energies_discarded,
                pretext=wfpretext,
                file_name=f"{filename_optimized}_dis_out.wf",
            )

            # optimize wavefunction
            cp(f"../{final_ami}.ami", "last.ami")
            # submit job
            self.print_job_file(
                self.partition,
                "last",
                self.n_tasks,
                "last",
            )
            run("sbatch amolqc_job")
            # check if job done
            job_done = False
            while not job_done:
                job_done = self.check_job_done("last")
                if not job_done:
                    if self.verbose:
                        print("job not done yet.")
                    time.sleep(20)

            last_wavefunction = self.get_final_wavefunction("last")

            cp(f"{last_wavefunction}.wf", f"../{dir_name}.wf")
            rm(f"{final_ami}.ami")
            if self.verbose:
                print()
                print("finish final block.")

    def do_initial_iteration(self, block_label: str, initial_ami: str):
        """"""
        dir_name = f"it{block_label}"
        mkdir(dir_name)

        with cd(dir_name):
            cp(f"../{self.wavefunction_name}.wf", ".")
            initial_determinant = self.sCI.build_energy_lowest_detetminant(
                self.N
            )
            self.sCI.get_initial_wf(
                self.S,
                self.M_s,
                self.n_MO,
                initial_determinant,
                self.excitations,
                self.orbital_symmetry,
                self.point_group,
                self.frozen_electrons,
                self.frozen_MOs,
                self.wavefunction_name,
                sort_option=self.sort_option,
                verbose=self.verbose,
            )
            # extract total number of csfs
            rm(f"{self.wavefunction_name}.wf")
            mv(
                f"{self.wavefunction_name}_out.wf",
                f"{self.wavefunction_name}.wf",
            )
            cp(f"../{initial_ami}.ami", ".")
            # submit job
            self.print_job_file(
                self.partition,
                dir_name,
                self.n_tasks,
                initial_ami,
            )
            run("sbatch amolqc_job")
            # check if job done
            job_done = False
            while not job_done:
                job_done = self.check_job_done(initial_ami)
                if not job_done:
                    if self.verbose:
                        print("job not done yet.")
                    time.sleep(20)
            # get last wavefunction and copy to folder with all blocks
            last_wavefunction = self.get_final_wavefunction(initial_ami)
            cp(f"{last_wavefunction}.wf", f"../{dir_name}.wf")
            cp(f"{self.wavefunction_name}_res.wf", f"../{dir_name}_res.wf")
        if self.verbose:
            print("finish initial block.")

    def do_selective_iteration(
        self,
        n_blocks: int,
        input_wf: str,
        iteration_ami: str,
        energy_ami: str,
        reference_determinant: list,
        excitations: list,
        excitations_on_ini: list,
    ):
        """"""
        if self.verbose:
            print(f"number of selections {n_blocks}")
        n_it = 0
        last_wavefunction = input_wf
        excitations_on = excitations_on_ini
        for i in range(n_blocks):
            print()
            print(f"Block iteration: {i+1}/{n_blocks}")
            print()
            n_it += 1
            dir_name = f"it{n_it}"
            mkdir(dir_name)
            # number corresponds to n-tuple excitation
            # get wavefunctions from previous iterations
            with cd(dir_name):
                try:
                    mv(f"../{last_wavefunction}_dis.wf", ".")
                except FileNotFoundError:
                    pass
                try:
                    mv(f"../{last_wavefunction}_nrg.amo", ".")
                except FileNotFoundError:
                    pass
                cp(f"../{last_wavefunction}.wf", ".")
                done = self.sCI.select_and_do_excitations(
                    self.N,
                    self.n_MO,
                    self.S,
                    self.M_s,
                    reference_determinant,
                    excitations,
                    excitations_on,
                    self.orbital_symmetry,
                    self.point_group,
                    self.frozen_electrons,
                    self.frozen_MOs,
                    last_wavefunction,
                    f"{last_wavefunction}_dis",
                    self.criterion,
                    self.threshold,
                    self.max_csfs,
                    threshold_type=self.threshold_type,
                    verbose=self.verbose,
                )
                mv(
                    f"{last_wavefunction}_out.wf",
                    f"{self.wavefunction_name}.wf",
                )
                mv(
                    f"{last_wavefunction}_dis_out.wf",
                    f"{self.wavefunction_name}_dis.wf",
                )
                cp(f"../{iteration_ami}.ami", ".")
                # submit job
                self.print_job_file(
                    self.partition,
                    dir_name,
                    self.n_tasks,
                    iteration_ami,
                )
                run("sbatch amolqc_job")
                job_done = False
                while not job_done:
                    job_done = self.check_job_done(iteration_ami)
                    if not job_done:
                        if self.verbose:
                            print("job not done yet.")
                        time.sleep(20)
                    # get last wavefunction and copy to folder with all blocks
                optimized_wavefunction = self.get_final_wavefunction(
                    iteration_ami
                )

                # compute energy criterion if required
                if self.criterion == "energy":
                    mv(f"{self.wavefunction_name}.wf", "tmp")
                    mv(
                        f"{optimized_wavefunction}.wf",
                        f"{self.wavefunction_name}.wf",
                    )
                    cp(f"../{energy_ami}.ami", ".")
                    self.print_job_file(
                        self.partition,
                        f"e_{n_it}",
                        self.n_tasks,
                        energy_ami,
                    )
                    run("sbatch amolqc_job")
                    # check if job done
                    job_done = False
                    while not job_done:
                        job_done = self.check_job_done(energy_ami)
                        if not job_done:
                            if self.verbose:
                                print("job not done yet.")
                            time.sleep(20)
                    mv(
                        f"{self.wavefunction_name}.wf",
                        f"{optimized_wavefunction}.wf",
                    )
                    mv("tmp", f"{self.wavefunction_name}.wf")
                    cp(f"{energy_ami}.amo", f"../{dir_name}_nrg.amo")
                    rm(f"{energy_ami}.ami")

                # copy results to folder with all blocks
                cp(f"{optimized_wavefunction}.wf", f"../{dir_name}.wf")
                cp(
                    f"{self.wavefunction_name}_dis.wf",
                    f"../{dir_name}_dis.wf",
                )
                last_wavefunction = dir_name
                excitations_on = [i + 1 for i in excitations_on]

                if done:
                    break

        if self.verbose:
            print("finish selective iteration")
        return last_wavefunction

    def blockwise_optimization(
        self,
        initial_ami,
        blockwise_ami,
        final_ami,
        energy_ami="",
        max_blocks=1000,
    ):
        #
        # initial block with adding jastrow and optimizing jastrow
        #
        n_block = "_initial"
        self.do_initial_block(n_block, initial_ami, energy_ami=energy_ami)
        # perform blockwise iterations
        #
        self.n_all_csfs = self.get_n_all_csfs("block_initial")
        n_blocks = math.ceil(
            (self.n_all_csfs - self.blocksize) / (self.blocksize - self.n_min)
        )
        if self.threshold_type == "sum_up":
            n_blocks = max_blocks
        flex = True
        if flex:
            n_blocks = max_blocks

        # n_blocks = math.ceil(174 / (self.blocksize - self.n_min))
        # blockwise iteration
        last_block = self.do_block_iteration(
            n_blocks, "block_initial", blockwise_ami, energy_ami=energy_ami
        )
        # final block
        self.do_final_block("final", f"{last_block}", final_ami)

        # sCI.select_and_do_next_package("discarded", wavefunction_name, "residual", threshold_ci, split_at=split_at, n_min=n_min, verbose=True)

    def do_iterative_construction(
        self,
        initial_ami,
        iteration_ami,
        final_ami,
        reference_determinant,
        energy_ami="",
    ):
        """"""
        self.do_initial_block("_ini", initial_ami, energy_ami=energy_ami)
        last_it = self.do_selective_iteration(
            4,
            "block_ini",
            iteration_ami,
            energy_ami,
            reference_determinant,
            [1],
            self.excitations,
        )
        # final block
        # self.do_final_block("final", f"{last_it}", final_ami)
