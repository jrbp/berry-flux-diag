#!/usr/bin/env python
from collections import MutableSequence
from itertools import count
from cmath import log
import numpy as np
from numba import jit


class EigenV():
    """
    Args:
        occupation: the occupation of the state
        gvecs: array of G-vectors
        evec: array of plane wave coefficients
              of the form [[c1_real, c1_imaginary],[c2_real, c2_imaginary],...]
              evec[i] corresponds to gvecs[i]
    """
    def __init__(self, occupation, gvecs, evec):
        self._occupation = occupation
        self._gvecs = gvecs
        self._evec = evec

    @property
    def occupation(self):
        return self._occupation

    @property
    def gvecs(self):
        return self._gvecs

    @property
    def evec(self):
        return self._evec

    def get_evec_complex(self):
        "get evec as a 1d complex vector"
        return self.evec[:, 0] + self.evec[:, 1]*1.j


class Kpoint(MutableSequence):
    """
    Acts as a list of EigenVs while also storing coordinates,
        weight, and the number of plane waves
    Args:
        kcoords: the (reduced) coordinates in reciprocal space
        weight: the weight of this kpoint
        planewaves: number of planewaves at this kpoint
        eigenvs: list of EigenV objects at this kpoint
    """
    def __init__(self, kcoords, weight, planewaves, eigenvs=None):
        self._kcoords = kcoords
        self._weight = weight
        self._planewaves = planewaves
        if eigenvs is not None:
            self._eigenvs = eigenvs
        else:
            self._eigenvs = []

    @property
    def eigenvs(self):
        return self._eigenvs

    @property
    def kcoords(self):
        return self._kcoords

    @property
    def weight(self):
        return self._weight

    @property
    def planewaves(self):
        return self._planewaves

    def __getitem__(self, ind):
        return self.eigenvs[ind]

    def __len__(self):
        return len(self.eigenvs)

    def __delitem__(self, i):
        self._eigenvs.__delitem__(i)

    def insert(self, i, eigenv):
        if isinstance(eigenv, EigenV):
            self._eigenvs.insert(i, eigenv)
        else:
            raise TypeError("Elements of Kpoint must be EigenV objects")

    def __setitem__(self, i, eigenv):
        if isinstance(eigenv, EigenV):
            self._eigenvs[i] = eigenv
        else:
            raise TypeError("Elements of Kpoint must be EigenV objects")

    def get_occupied_only(self):
        """returns a Kpoint object which only contains EigenV objects with
               occupation > 0.9"""
        return Kpoint(self.kcoords, self.weight, self.planewaves,
                      [s for s in self.eigenvs if s.occupation > 0.9])

    def get_g_shifted(self, g_shift):
        """returns a Kpoint object where gvecs at all eigenvs have been modified
        such that the returned Kpoint represents the Kpoint at self.kcoords + g_shift"""
        return Kpoint(tuple(np.array(self.kcoords) + np.array(g_shift)), self.weight, self.planewaves,
                      [EigenV(s.occupation,
                              np.array([np.array(g) - np.array(g_shift) for g in s.gvecs]),
                              s.evec)
                       for s in self.eigenvs])


def sort_eigenv(eigenv, kcoord=[0., 0., 0.]):
    """returns an EigenV object with components sorted by magnitude of the gvector+kcoord
    No longer needed since overlaps check g vector correspondence exactly"""
    gv_ev = list(zip(eigenv.gvecs, eigenv.evec))
    sorted_gv_ev = sorted(gv_ev,
                          key=(lambda x:
                               np.linalg.norm(np.array(x[0]) + np.array(kcoord))))
    sorted_gv = np.array([x[0] for x in sorted_gv_ev])
    sorted_ev = np.array([x[1] for x in sorted_gv_ev])
    sorted_eigenv = EigenV(eigenv.occupation, sorted_gv, sorted_ev)
    return sorted_eigenv


def read_wfc(wfc_file, return_first_k=False, sort_by_gvec_mag=False):
    """
    Args:
        wfc_file: file to read in (from output of modified cut3d)
        return_first_k : for testing only, only returns a single Kpoint
    returns:
        list of Kpoints
    """
    kpoints = []
    with open(wfc_file, 'r', 1) as f:
        for line in f:
            (max_planewaves, pws_in_calc, weight,
                kx, ky, kz, occ) = [float(x) for x in line.split()]
            # IMPORTANT:  below change planewaves between the following two lines
            #             if using the fft to make same number of planewaves at each kpt
            #             this is due to how the modified cut3d is writing wfcs, and should be changed there
            planewaves = int(max_planewaves)  # if fft used
            # planewaves = int(pws_in_calc)   # if fft not used
            try:
                # if this_kpoint.kcoords != (kx, ky, kz): # should subtract for comparison of floats
                if (np.abs(np.array(this_kpoint.kcoords) - np.array([kx, ky, kz])) > 1.e-5).any():
                    print("finished reading kpoint "
                          "{}".format(this_kpoint.kcoords))
                    if return_first_k:
                        return this_kpoint
                    kpoints.append(this_kpoint)
                    this_kpoint = Kpoint((kx, ky, kz), weight, planewaves)
            except NameError:
                this_kpoint = Kpoint((kx, ky, kz), weight, planewaves)
            gvecs = np.zeros((planewaves, 3), dtype=int)
            evec = np.zeros((planewaves, 2), dtype=float)
            for i in range(planewaves):
                (gvecs[i, 0], gvecs[i, 1], gvecs[i, 2],
                    evec[i, 0], evec[i, 1]) = f.readline().split()
            if sort_by_gvec_mag:
                this_kpoint.append(sort_eigenv(EigenV(occ, gvecs, evec)))
            else:
                this_kpoint.append(EigenV(occ, gvecs, evec))
    print("finished reading kpoint {}".format(this_kpoint.kcoords))
    kpoints.append(this_kpoint)
    return kpoints


def compute_overlap_same_gvecs(evs0, evs1):
    """
    fast, but only works when sets of evs have exact same gvecs in the same order
    """
    overlap = np.zeros((len(evs0), len(evs0)), dtype=complex)
    for m, ev0 in enumerate(evs0):
        for n, ev1 in enumerate(evs1):
            overlap[m, n] = np.vdot(ev0.get_evec_complex(),
                                    ev1.get_evec_complex())
    return overlap


def compute_overlap(evs0, evs1, dg=np.array([0, 0, 0])):
    """
    Args: evs0, evs1: each a list of EigenV objects
    Returns: overlap matrix overlap[m,n] = <evs0_m | evs1_n>
    """
    overlap = np.zeros((len(evs0), len(evs0)), dtype=complex)
    for m, ev0 in enumerate(evs0):
        for n, ev1 in enumerate(evs1):
            # # slow non jit way
            # this_element = 0.
            # for gvec0, ipw0 in zip(ev0.gvecs, np.conj(ev0.get_evec_complex())):
            #     for gvec1, ipw1 in zip(ev1.gvecs, ev1.get_evec_complex()):
            #         if all(gvec0 == gvec1):
            #             this_element += ipw0 * ipw1
            overlap[m, n] = compute_overlap_element(ev0.gvecs, np.conj(ev0.get_evec_complex()),
                                                    ev1.gvecs, ev1.get_evec_complex(), dg)
    return overlap


@jit(nopython=True, cache=True)
def compute_overlap_element(gvs0, ev0_conj, gvs1, ev1, dg):
    this_element = 0.
    for i in range(len(gvs0)):
        for j in range(len(gvs1)):
            if (gvs0[i][0] == gvs1[j][0] - dg[0]
               and gvs0[i][1] == gvs1[j][1] - dg[1]
               and gvs0[i][2] == gvs1[j][2] - dg[2]):
                this_element += ev0_conj[i] * ev1[j]
    return this_element


def find_min_singular_value(wfc0, wfc1, same_gvecs=False):
    """
    Args: wfc0, wfc1: each a list of Kpoint objects
    Returns: The (value, coordinate) of the smallest of all singular values
    across all kpoints from the singular value decomposition
    preformed at each kpoint, if this is too small the states
    were unable to be reasonably aligned
    """
    s_mins = np.zeros(len(wfc0))
    for i, kpt0, kpt1 in zip(count(), wfc0, wfc1):
        if same_gvecs:
            overlap = compute_overlap_same_gvecs(kpt0.get_occupied_only(),
                                                 kpt1.get_occupied_only())
        else:
            overlap = compute_overlap(kpt0.get_occupied_only(),
                                      kpt1.get_occupied_only())
        s = np.linalg.svd(overlap, compute_uv=False)
        s_mins[i] = s.min()
    return s_mins.min(), wfc0[s_mins.argmin()].kcoords


def bphase_along_string(kpt_string, cheap_pt=False):
    tot_phase_change = 0.
    for i in range(len(kpt_string)):
        if cheap_pt:
            if i == len(kpt_string) - 1:
                fromkpt = kpt_string[i].get_occupied_only()
                tokpt = kpt_string[0].get_occupied_only()
                overlap = compute_overlap(fromkpt, tokpt,
                                          dg=np.array([0, 0, 1]))
            else:
                fromkpt = kpt_string[i].get_occupied_only()
                tokpt = kpt_string[i + 1].get_occupied_only()
                raw_overlap = compute_overlap(fromkpt, tokpt)
                u, s, v = np.linalg.svd(raw_overlap)
                overlap = np.dot(u, v)
        else:
            if i == len(kpt_string) - 1:
                fromkpt = kpt_string[i].get_occupied_only()
                tokpt = kpt_string[0].get_occupied_only()
                overlap = compute_overlap(fromkpt,
                                          tokpt,
                                          dg=np.array([0, 0, 1]))

                # # slower, test of get_g_shifted method
                # tokpt = kpt_string[0].get_g_shifted([0, 0, 1])
                # overlap = compute_overlap(fromkpt.get_occupied_only(),
                #                           tokpt.get_occupied_only())
            else:
                fromkpt = kpt_string[i]
                tokpt = kpt_string[i + 1]
                overlap = compute_overlap(kpt_string[i].get_occupied_only(),
                                          kpt_string[i+1].get_occupied_only())
        # this_det = np.linalg.det(overlap)
        # phase_change = log(this_det).imag
        s, lndet = np.linalg.slogdet(overlap)
        # print(s, lndet)
        phase_change = log(s).imag
        print(fromkpt.kcoords, "->",
              tokpt.kcoords,
              " ", 2 * phase_change / (2 * np.pi))
        tot_phase_change += phase_change
    return tot_phase_change


def bphase_with_mult(kpt_string, cheap_pt=False):
    product = np.identity(len(kpt_string[0].get_occupied_only()), dtype=complex)
    for i in range(len(kpt_string)):
        if cheap_pt:
            if i == len(kpt_string) - 1:
                fromkpt = kpt_string[i].get_occupied_only()
                tokpt = kpt_string[0].get_occupied_only()
                raw_overlap = compute_overlap(fromkpt, tokpt,
                                              dg=np.array([0, 0, 1]))
            else:
                fromkpt = kpt_string[i].get_occupied_only()
                tokpt = kpt_string[i + 1].get_occupied_only()
                raw_overlap = compute_overlap(fromkpt, tokpt)
            u, s, v = np.linalg.svd(raw_overlap)
            overlap = np.dot(u, v)
        else:
            if i == len(kpt_string) - 1:
                overlap = compute_overlap(kpt_string[i].get_occupied_only(),
                                          kpt_string[0].get_occupied_only(),
                                          dg=np.array(np.array([0, 0, 1])))
            else:
                overlap = compute_overlap(kpt_string[i].get_occupied_only(),
                                          kpt_string[i+1].get_occupied_only())
        product = np.dot(product, overlap)
    s, lndet = np.linalg.slogdet(product)
    return log(s).imag


def det_of_string_mat_mult(kpt_string):
    product = np.identity(len(kpt_string[0].get_occupied_only()), dtype=complex)
    for i in range(len(kpt_string)):
        if i == len(kpt_string) - 1:
            overlap = compute_overlap(kpt_string[i].get_occupied_only(),
                                      kpt_string[0].get_occupied_only(),
                                      dg=np.array([0, 0, 1]))
        else:
            overlap = compute_overlap(kpt_string[i].get_occupied_only(),
                                      kpt_string[i+1].get_occupied_only())
        product = np.dot(product, overlap)
    # s, lndet = np.linalg.slogdet(product)   # Do I need to be careful about numerics here?
    # return s * np.exp(lndet)
    return np.linalg.det(product)


def det_of_string(kpt_string):
    det_product = 1. + 0.j
    # set_of_overlaps = ""
    for i in range(len(kpt_string)):
        if i == len(kpt_string) - 1:
            overlap = compute_overlap(kpt_string[i].get_occupied_only(),
                                      kpt_string[0].get_occupied_only(),
                                      dg=np.array([0, 0, 1]))

            # # slower, test of get_g_shifted method
            # overlap = compute_overlap(kpt_string[i].get_occupied_only(),
            #                           kpt_string[0].get_g_shifted([0, 0, 1]).get_occupied_only())
            # set_of_overlaps += str(kpt_string[i].kcoords) + '-->' + str(kpt_string[0].kcoords) + '\n'
        else:
            overlap = compute_overlap(kpt_string[i].get_occupied_only(),
                                      kpt_string[i+1].get_occupied_only())
            # set_of_overlaps += str(kpt_string[i].kcoords) + '-->' + str(kpt_string[i+1].kcoords) + '\n'
        determinant = np.linalg.det(overlap)
        det_product *= determinant
    # print(set_of_overlaps, '\n')
    return det_product


def phase_string_true_pt(kpt_string):
    from_kpt = kpt_string[0].get_occupied_only()
    tot_phase_change = 0.
    for i in range(len(kpt_string)):
        if i == len(kpt_string) - 1:
            to_kpt = kpt_string[0].get_g_shifted([0, 0, 1]).get_occupied_only()
        else:
            to_kpt = get_kpoint2_aligned_with_kpoint1(from_kpt, kpt_string[i+1].get_occupied_only())
        overlap = compute_overlap(from_kpt, to_kpt)
        s, lndet = np.linalg.slogdet(overlap)
        phase_change = log(s).imag
        print(from_kpt.kcoords, "->",
              to_kpt.kcoords,
              " ", 2 * phase_change / (2 * np.pi))
        tot_phase_change += phase_change
        from_kpt = to_kpt
    return tot_phase_change


def get_string(wfc, kx, ky):
    result = []
    for kpt in wfc:
        in_this_string = ((abs(kpt.kcoords[0] - kx) < 1.e-5)
                          and (abs(kpt.kcoords[1] - ky) < 1.e-5))
        if in_this_string:
            result.append(kpt)
    result.sort(key=lambda k: k.kcoords[-1])
    return result


def get_string_indicies(wfc, kx, ky):
    result = []
    for i, kpt in enumerate(wfc):
        in_this_string = ((abs(kpt.kcoords[0] - kx) < 1.e-5)
                          and (abs(kpt.kcoords[1] - ky) < 1.e-5))
        if in_this_string:
            result.append((i, kpt))
    result.sort(key=lambda k: k[1].kcoords[-1])
    return [entry[0] for entry in result]


def get_berry_phase_polarization(wfc, method=None):
    bz_2d_points = []
    for kpt in wfc:
        bz_2d_points.append((kpt.kcoords[0], kpt.kcoords[1]))

    nstr = len(set(bz_2d_points))

    if method is None or method == 'det_avg':
        # nearly an exact copy of what abinit does
        # can not see how phase evolves along string
        det_strings = []
        string_coords = []
        det_avg = 0
        for kx, ky in set(bz_2d_points):
            print("computing phase for string {}, {}".format(kx, ky))
            this_string = get_string(wfc, kx, ky)
            string_coords.append((kx, ky))
            this_det = det_of_string(this_string)
            # this_det = det_of_string_mat_mult(this_string)
            det_strings.append(this_det)
            det_avg += this_det / nstr

            phase0 = np.arctan2(det_avg.imag, det_avg.real)
            # phase0 = np.log(det_avg).imag
            det_mod = np.conjugate(det_avg) * det_avg

            polb = 0.
            polb_str = []
            for det_string in det_strings:
                rel_string = (np.conj(det_avg) * det_string) / det_mod
                dphase = np.arctan2(rel_string.imag, rel_string.real)
                # dphase = np.log(rel_string).imag
                this_polb = 2*(phase0 + dphase) / (2 * np.pi)
                polb_str.append(this_polb)
                # print(this_polb)
                polb += this_polb / float(nstr)
                # print(nstr)
        # for kpt, phase, in zip(string_coords, polb_str):
        #     print(kpt, phase)

    elif method == "true_pt":
        bps = []
        for kx, ky in set(bz_2d_points):
            print()
            this_string = get_string(wfc, kx, ky)
            bp = phase_string_true_pt(this_string)
            bps.append(bp)
            print(kx, "\t", ky, '\t\t', (2 * bp) / (2*np.pi))
        polb = 2 * sum(bps) / (2 * np.pi * len(bps))
    else:
        bps = []
        for kx, ky in set(bz_2d_points):
            print()
            this_string = get_string(wfc, kx, ky)
            # bp_s = bphase_along_string(this_string, cheap_pt=False)
            # bp = bphase_with_mult(this_string, cheap_pt=False)
            bp = bphase_along_string(this_string, cheap_pt=False)
            bps.append(bp)
            # print(kx, "\t", ky, '\t\t', (2 * bp) / (2*np.pi), (2 * bp_s) / (2*np.pi))
            print(kx, "\t", ky, '\t\t', (2 * bp) / (2*np.pi))
        polb = 2 * sum(bps) / (2 * np.pi * len(bps))
    return polb


def get_kpoint2_aligned_with_kpoint1(kpoint1, kpoint2):
    """returns a Kpoint object corresponding to kpoint2
    with a unitary operation applied such that bands
    are maximally aligned with kpoint1
    """
    raw_overlap = compute_overlap(kpoint1, kpoint2)
    u, s, v = np.linalg.svd(raw_overlap)
    print("aligning: \n kpoint1: {} \n kpoint2: {} \n min singular value: {}".format(
        kpoint1.kcoords, kpoint2.kcoords, s.min()))
    rot_mat = np.linalg.inv(np.dot(u, v))
    new_eigenvs = np.zeros((len(kpoint2), len(kpoint2[0].evec), 2))
    for i in range(len(kpoint2)):
        for j in range(len(kpoint2)):
            # TODO: Should really just make EigenV store the complex number directly
            new_eigenvs[i, :, 0] += (rot_mat[i, j].real * kpoint2[j].evec[:, 0]
                                     - rot_mat[i, j].imag * kpoint2[j].evec[:, 1])
            new_eigenvs[i, :, 1] += (rot_mat[i, j].real * kpoint2[j].evec[:, 1]
                                     + rot_mat[i, j].imag * kpoint2[j].evec[:, 0])
    # TODO: maybe change things so we don't need to put the old occupations here, really
    #       this function should only be called with Kpoint objects where all bands
    #       are occupied, if this isn't the case the occupations assigned below are nonsense
    new_eigenv_objects = [EigenV(kpoint2[n].occupation, kpoint2[n].gvecs, ev)
                          for n, ev in enumerate(new_eigenvs)]
    new_kpt = Kpoint(kpoint2.kcoords, kpoint2.weight,
                     kpoint2.planewaves, eigenvs=new_eigenv_objects)
    return new_kpt


def get_wfc2_aligned_with_wfc1(wfc1, wfc2):
    """returns a list of Kpoints object corresponding to wfc2
    with a unitary operation applied to each Kpoint such that bands
    are maximally aligned with corresponding Kpoints in wfc1
    kpoints must be in the same order"""
    return [get_kpoint2_aligned_with_kpoint1(kp1, kp2) for kp1, kp2 in zip(wfc1, wfc2)]


def construct_pt_gauge_along_kz(wfc):
    smooth_wfc = [{}]*len(wfc)
    bz_2d_points = []
    for kpt in wfc:
        bz_2d_points.append((kpt.kcoords[0], kpt.kcoords[1]))
    for kx, ky in set(bz_2d_points):
        print(kx, ky)
        this_string_indicies = get_string_indicies(wfc, kx, ky)
        first_point = True
        for i in this_string_indicies:
            this_kpt = wfc[i].get_occupied_only()
            if first_point:
                smooth_wfc[i] = this_kpt
                last_point_ind = i
                first_point = False
            else:
                this_kpt_aligned = get_kpoint2_aligned_with_kpoint1(
                    smooth_wfc[last_point_ind], this_kpt)
                smooth_wfc[i] = this_kpt_aligned
                last_point_ind = i
    return smooth_wfc


def get_indiv_band_bphases(kpoint1, kpoint2, returnEigVecs=False):
    raw_overlap = compute_overlap(kpoint1, kpoint2)
    u, s, v = np.linalg.svd(raw_overlap)
    # rot_mat = np.linalg.inv(np.dot(u, v))
    rot_mat = np.dot(u, v)
    rot_mat_eigenvals = np.linalg.eigvals(rot_mat)
    if returnEigVecs:
        eigvecs = np.linalg.eig(rot_mat)[1]
        return np.log(rot_mat_eigenvals).imag, eigvecs
    else:
        return np.log(rot_mat_eigenvals).imag


def get_indiv_band_bphase_differences(wAkpt1, wAkpt2, wBkpt1, wBkpt2, nspin=1):
    """
    Computes individual band berry phases for state A from wAkpt1 to wAkpt2
    then computes individual band berry phases for state B from wBkpt1 to wBkpt2
    then returns an array of the smallest differences between corresponding indiv band phases
    THIS IS NONSENSE SINCE INDIV BAND BPHASES ARE IN ARBITRARY ORDER
    """
    wA_phases = get_indiv_band_bphases(wAkpt1, wAkpt2)
    wB_phases = get_indiv_band_bphases(wBkpt1, wBkpt2)

    #  testing, looking at overlap of eigenvecs
    #print("getting phases for A")
    #wA_phases, wA_vecs = get_indiv_band_bphases(wAkpt1, wAkpt2, returnEigVecs=True)
    #print("getting phases for B")
    #wB_phases, wB_vecs = get_indiv_band_bphases(wBkpt1, wBkpt2, returnEigVecs=True)
    #overlap = np.zeros((len(wA_vecs), len(wA_vecs)), dtype=complex)
    #for m, ev0 in enumerate(wA_vecs):
    #    for n, ev1 in enumerate(wB_vecs):
    #        overlap[m, n] = np.vdot(ev0, ev1) * np.conj(np.vdot(ev0, ev1))
    #print(overlap)

    changes = (wB_phases - wA_phases)
    print('A phases')
    print(wA_phases / np.pi)
    print('A sum = {}'.format(sum(wA_phases) / np.pi))
    print('B phases')
    print(wB_phases / np.pi)
    print('B sum = {}'.format(sum(wB_phases) / np.pi))
    min_changes = changes - (2 * np.pi) * np.round((changes / (2 * np.pi))).astype(int)
    return min_changes


def get_phase_difference_from_strings(kpt_string1, kpt_string2):
    """
    Given two kpoint strings compute differences in individual band berry phases for each
    pair of kpts in a string

    Should only be useful if kpt_string2 is aligned with kpt_string1
    (same set of kpts and appropriate gauge choice)
    not currently enforcing any of this for testing purposes, use with caution
    """
    tot_phase_difference = 0.
    for i in range(len(kpt_string1)):
        if i == len(kpt_string1) - 1:
            fromkptA = kpt_string1[i]
            fromkptB = kpt_string2[i]
            tokptA = kpt_string1[0].get_g_shifted([0, 0, 1])
            tokptB = kpt_string2[0].get_g_shifted([0, 0, 1])
        else:
            fromkptA = kpt_string1[i]
            fromkptB = kpt_string2[i]
            tokptA = kpt_string1[i+1]
            tokptB = kpt_string2[i+1]
        print("{} -> {}".format(fromkptA.kcoords, tokptA.kcoords))
        phase_differences = get_indiv_band_bphase_differences(fromkptA, tokptA,
                                                              fromkptB, tokptB)
        print('B - A')
        print(phase_differences / np.pi)
        tot_phase_difference += sum(phase_differences)
        print('sum for this {} -> {}'.format(fromkptA.kcoords, tokptA.kcoords))
        print(sum(phase_differences) / np.pi)
    return tot_phase_difference


def get_phase_difference_from_wfcs(wfc1, wfc2):
    bz_2d_points = []
    for kpt in wfc1:
        bz_2d_points.append((kpt.kcoords[0], kpt.kcoords[1]))

    bz_2d_set = set(bz_2d_points)
    phase_diffs = []
    for kx, ky in bz_2d_set:
        string1 = get_string(wfc1, kx, ky)
        string2 = get_string(wfc2, kx, ky)
        this_ph_diff = get_phase_difference_from_strings(string1, string2)
        phase_diffs.append(this_ph_diff)
        print()
        print(kx, "\t", ky, '\t\t')
        print("\t", (2 * this_ph_diff) / (2*np.pi))
        print()
    avg_phase_diff = 2 * sum(phase_diffs) / (2 * np.pi * len(phase_diffs))
    return avg_phase_diff


def testing_indiv_bphase_along_string(kpt_string):
    tot_phase_change = 0.
    for i in range(len(kpt_string)):
        fromkpt = kpt_string[i].get_occupied_only()
        if i == len(kpt_string) - 1:
            tokpt = kpt_string[0].get_g_shifted([0, 0, 1]).get_occupied_only()
        else:
            tokpt = kpt_string[i + 1].get_occupied_only()
        overlap = compute_overlap(fromkpt, tokpt)
        # s, lndet = np.linalg.slogdet(overlap)
        # phase_change = log(s).imag
        phase_change = np.log(np.linalg.det(overlap)).imag
        print(fromkpt.kcoords, "->", tokpt.kcoords)
        print('\t', 'phase change from raw_overlap: {}'.format(phase_change / np.pi))
        u, s, v = np.linalg.svd(overlap)
        # rot_mat = np.linalg.inv(np.dot(u, v))
        rot_mat = np.dot(u, v)
        rot_mat_eigenvals_ph = np.imag(np.log(np.linalg.eigvals(rot_mat)))
        print('\t', 'sum of new rot_mat eigenval phases: {}'.format(rot_mat_eigenvals_ph.sum() / np.pi))
        print('\t', 'phase of det of rot_mat: {}'.format(np.log(np.linalg.det(rot_mat)).imag / np.pi))
        print('\t', 'new rot_mat eigenval phases: {}'.format(rot_mat_eigenvals_ph / np.pi))
        tot_phase_change += phase_change
    return tot_phase_change


def testing_indiv_get_kpoint2_aligned_with_kpoint1(kpoint1, kpoint2):
    raw_overlap = compute_overlap(kpoint1, kpoint2)
    u, s, v = np.linalg.svd(raw_overlap)
    print("aligning: \n kpoint1: {} \n kpoint2: {} \n min singular value: {}".format(
        kpoint1.kcoords, kpoint2.kcoords, s.min()))
    rot_mat = np.linalg.inv(np.dot(u, v))
    rot_mat_eigenvals_ph = np.imag(np.log(np.linalg.eigvals(rot_mat)))
    print('\t', 'sum of new rot_mat eigenval phases: {}'.format(rot_mat_eigenvals_ph.sum() / np.pi))
    print('\t', 'phase of det of rot_mat: {}'.format(np.log(np.linalg.det(rot_mat)).imag / np.pi))
    print('\t', 'new rot_mat eigenval phases: {}'.format(rot_mat_eigenvals_ph / np.pi))
    new_eigenvs = np.zeros((len(kpoint2), len(kpoint2[0].evec), 2))
    for i in range(len(kpoint2)):
        for j in range(len(kpoint2)):
            # TODO: Should really just make EigenV store the complex number directly
            new_eigenvs[i, :, 0] += (rot_mat[i, j].real * kpoint2[j].evec[:, 0]
                                     - rot_mat[i, j].imag * kpoint2[j].evec[:, 1])
            new_eigenvs[i, :, 1] += (rot_mat[i, j].real * kpoint2[j].evec[:, 1]
                                     + rot_mat[i, j].imag * kpoint2[j].evec[:, 0])
    # TODO: maybe change things so we don't need to put the old occupations here, really
    #       this function should only be called with Kpoint objects where all bands
    #       are occupied, if this isn't the case the occupations assigned below are nonsense
    new_eigenv_objects = [EigenV(kpoint2[n].occupation, kpoint2[n].gvecs, ev)
                          for n, ev in enumerate(new_eigenvs)]
    new_kpt = Kpoint(kpoint2.kcoords, kpoint2.weight,
                     kpoint2.planewaves, eigenvs=new_eigenv_objects)
    return new_kpt


if __name__ == '__main__':
    import sys

    print("reading {}".format(sys.argv[1]))
    wfc0 = read_wfc(sys.argv[1])

    print("reading {}".format(sys.argv[2]))
    wfc1 = read_wfc(sys.argv[2])

    wfc_occ0 = [k.get_occupied_only() for k in wfc0]
    wfc_occ1 = [k.get_occupied_only() for k in wfc1]

    # wfc_occ0_smooth = construct_pt_gauge_along_kz(wfc_occ0)

    # wfc_occ1_aligned = get_wfc2_aligned_with_wfc1(wfc_occ0_smooth, wfc_occ1)
    wfc_occ1_aligned = get_wfc2_aligned_with_wfc1(wfc_occ0, wfc_occ1)

    # wfc0_no_pt_s = get_string(wfc_occ0, -.3333, 0.)
    # wfc0_pt_s = get_string(wfc_occ0_smooth, -.3333, 0.)
    # # wfc1_no_align = get_string(wfc_occ1, -.3333, 0.)
    # wfc1_aligned = get_string(wfc_occ1_aligned, -.3333, 0.)

    # phase_changes = get_indiv_band_bphase_differences(wfc0_pt_s[0], wfc0_pt_s[1],
    #                                                   wfc1_aligned[0], wfc1_aligned[1])
    # phase_change = get_phase_difference_from_strings(wfc0_no_pt_s, wfc1_aligned)
    phase_change = get_phase_difference_from_wfcs(wfc_occ0, wfc_occ1_aligned)
    print(phase_change / np.pi)

    # # print('no pt wfc0')
    # # testing_indiv_bphase_along_string(wfc0_no_pt_s)
    # # print()
    # # print('pt wfc0')
    # testing_indiv_bphase_along_string(wfc0_pt_s)
    # # print()
    # # print('aligned wfc1')
    # # testing_indiv_bphase_along_string(wfc1_aligned)
    # # print()

    # # print('at {}'.format(wfc0_pt_s[0].kcoords))
    # # testing_indiv_get_kpoint2_aligned_with_kpoint1(wfc0_pt_s[0], wfc1_no_align[0])
    # # print('at {}'.format(wfc0_pt_s[1].kcoords))
    # # testing_indiv_get_kpoint2_aligned_with_kpoint1(wfc0_pt_s[1], wfc1_no_align[1])
    # # print('at {}'.format(wfc0_pt_s[2].kcoords))
    # # testing_indiv_get_kpoint2_aligned_with_kpoint1(wfc0_pt_s[2], wfc1_no_align[2])

    # # BEGIN BLOCK OF COMPARING TOTAL BPHASES WITH ALIGNMENT
    # bp0 = get_berry_phase_polarization(wfc_occ0_smooth, method='verbose')
    # print("berry phase wfc_occ0: {}".format(bp0))
    # # bp1 = get_berry_phase_polarization(wfc_occ1, method='verbose')
    # # print("berry phase wfc_occ1: {}".format(bp1))
    # bp1_aligned = get_berry_phase_polarization(wfc_occ1_aligned, method='verbose')
    # print("berry phase wfc_occ1_aligned: {}".format(bp1_aligned))

    # # print("bp1 - bp0 = {}".format(bp1 - bp0))
    # print("bp1_aligned - bp0 = {}".format(bp1_aligned - bp0))

    # # END BLOCK OF COMPARING TOTAL BPHASES WITH ALIGNMENT

    # bp0 = get_berry_phase_polarization(wfc0, method='verbose')
    # bp0 = get_berry_phase_polarization(wfc0)
    # bp0 = get_berry_phase_polarization(wfc0, method='true_pt')
    # print("Electronic berry phase: {}".format(bp0))

    ## kpt_a0 = wfc0[0].get_occupied_only()
    ## kpt_b0 = wfc0[36].get_occupied_only()
    # kpt_b0 = wfc0[0].get_occupied_only().get_g_shifted([0, 0, 1])

    ## print(kpt_a0.kcoords)
    ## print(kpt_b0.kcoords)

    ## overlap_ab0 = compute_overlap(kpt_a0, kpt_b0)
    ## phase_ab0 = np.log(np.linalg.det(overlap_ab0)).imag / np.pi
    ## print("phase a-b = {}".format(phase_ab0))
    ## u, s, v = np.linalg.svd(overlap_ab0)
    ## pt_overlap_ab0 = np.dot(u, v)
    ## pt_phase_ab0 = np.log(np.linalg.det(pt_overlap_ab0)).imag / np.pi
    ## print("cheap_pt phase a-b = {}".format(pt_phase_ab0))

    ## kpt_b0_rot = get_kpoint2_aligned_with_kpoint1(kpt_a0, kpt_b0)
    ## overlap_true_pt = compute_overlap(kpt_a0, kpt_b0_rot)
    ## phase_true_pt = np.log(np.linalg.det(overlap_true_pt)).imag / np.pi
    ## print("true_pt phase a-b = {}".format(phase_true_pt))


    ## # find kpt indices along 0,0
    ## for i, k in enumerate(wfc0):
    ##     if k.kcoords[0] == 0 and k.kcoords[1] == 0:
    ##         print(i, k.kcoords)
    # print()
    # for i, k in enumerate(wfc1):
    #     if k.kcoords[0] == 0 and k.kcoords[1] == 0:
    #         print(i, k.kcoords)

    # print("smallest singular value is "
    #       "{} at the point {}".format(*find_min_singular_value(wfc0, wfc1)))

    # bz_2d_points = []
    # for kpt in wfc0:
    #     bz_2d_points.append((kpt.kcoords[0], kpt.kcoords[1]))

    # string_vals = []
    # num_strings = len(set(bz_2d_points))
    # for kx, ky in set(bz_2d_points):
    #     print(kx, ", ", ky)
    #     val = compute_phase_diff_along_string(wfc0, wfc1, kx, ky)
    #     string_vals.append(val)
    #     print(val)
    #     print()
    # print(sum(string_vals)/num_strings)
