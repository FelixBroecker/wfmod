from csf import SelectedCI


class AddSingles:
    def __init__(self):
        self.sCI = SelectedCI()

    def hex2bin(self, hex, bit_len):
        """"""
        integer_value = int(hex, 16)
        binary_string = bin(integer_value)[2:]

        return binary_string.zfill(bit_len)

    def bin2det(self, bin, sgn=1):
        """"""
        return [
            sgn * (i + 1) for i, bit in enumerate(reversed(bin)) if bit == "1"
        ]

    def parse_cipsi_dets(self, filename: str):
        """
        parse the hexadecimal abbreviation and amplitudes
        in quantum package2 output.
        """
        n_mo = 0
        index = []
        determinant = []
        amplitude = []
        counter = 0
        found = False
        with open(filename, "r") as reffile:
            for line in reffile:
                counter += 1
                if "mo_num" in line:
                    n_mo = int(line.split()[-1])
                if "i =" in line:
                    index.append(int(line.split()[-1]))
                    found = True
                    counter = 0
                if "amplitude" in line:
                    amplitude.append(float(line.split()[-1]))
                if found and counter == 2:
                    determinant.append(line.replace("\n", "").split("|"))

        # get deterinant format
        determinant = self.parse_qp_dets(determinant, n_mo)
        return amplitude, determinant

    def parse_qp_dets(self, determinants, n_mo):
        """ """
        res = []
        for pair in determinants:
            bin = self.hex2bin(pair[0], n_mo)
            alpha = self.bin2det(bin, sgn=1)

            bin = self.hex2bin(pair[1], n_mo)
            beta = self.bin2det(bin, sgn=-1)
            res.append(alpha + beta)
        return res

    def cipsi2amolqc(self, S, M_s, wavefunction_name, wftyp, split_at):
        """"""
        # parse determinants and print them in AMOLQC format
        ci_coefficients, determinants = self.parse_cipsi_dets(
            f"{wavefunction_name}.wf"
        )
        if wftyp == "det" or wftyp == "det+csf":
            self.sCI.write_AMOLQC(
                [],
                determinants[:split_at],
                ci_coefficients[:split_at],
                pretext="",
                file_name=f"{wavefunction_name}_dets.wf",
                wftype="det",
            )
            print(len(determinants))

        # get csfs from determinant basis and print wavefunction.
        # create guess for CI coefficients
        if wftyp == "csf" or wftyp == "det+csf":
            csf_coefficients, csfs = self.sCI.get_unique_csfs(
                determinants, S, M_s
            )
            csf_coefficients, csfs = self.sCI.sort_determinants_in_csfs(
                csf_coefficients, csfs
            )
            ci_csf_coefficients = [
                1 if n == 0 else 0 for n in range(len(csfs))
            ]

            self.sCI.write_AMOLQC(
                csf_coefficients[:split_at],
                csfs[:split_at],
                ci_csf_coefficients[:split_at],
                pretext="",
                file_name=f"{wavefunction_name}_csfs.wf",
            )
            print(f"len csfs: {len(csfs)}")

            # expand again in determinants to see how may
            # determinants have been added
            _, _, determinant_basis_csfs = self.sCI.get_transformation_matrix(
                csf_coefficients, csfs, range(len(csf_coefficients))
            )
            print(
                f"re-expansion in determinants: {len(determinant_basis_csfs)}"
            )
            print(
                f"added determinants: \
                    {len(determinant_basis_csfs)-len(determinants)}"
            )

    # def read_cipsi(self, wavefunction_name, S, M_s, split_at):
    #     wf_name_praefix = wavefunction_name.split(".")[0]
    #     # parse determinants and print them in AMOLQC format
    #     ci_coefficients, determinants = utils.parse_cipsi_dets(
    #         wavefunction_name
    #     )
    #     sCI.write_AMOLQC(
    #         [],
    #         determinants[:split_at],
    #         ci_coefficients[:split_at],
    #         pretext="",
    #         file_name=f"{wf_name_praefix}_dets.wf",
    #         wftype="det",
    #     )
    #     print(len(determinants))

    #     # get csfs from determinant basis and print wavefunction.
    #     # create guess for CI coefficients
    #     csf_coefficients, csfs = sCI.get_unique_csfs(determinants, S, M_s)
    #     csf_coefficients, csfs = sCI.sort_determinants_in_csfs(
    #         csf_coefficients, csfs
    #     )
    #     ci_csf_coefficients = [1 if n == 0 else 0 for n in range(len(csfs))]

    #     sCI.write_AMOLQC(
    #         csf_coefficients[:split_at],
    #         csfs[:split_at],
    #         ci_csf_coefficients[:split_at],
    #         pretext="",
    #         file_name=f"{wf_name_praefix}_csfs.wf",
    #     )
    #     print(f"len csfs: {len(csfs)}")

    #     # expand again in determinants to see how may
    #     # determinants have been added
    #     _, _, determinant_basis_csfs = sCI.get_transformation_matrix(
    #         csf_coefficients, csfs, range(len(csf_coefficients))
    #     )
    #     print(len(determinant_basis_csfs))
