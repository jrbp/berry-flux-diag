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
        return Kpoint(np.array(self.kcoords) + np.array(g_shift), self.weight, self.planewaves,
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


def compute_overlap(evs0, evs1, dg=np.array([0, 0, 0])):
    """
    Args: evs0, evs1: each a list of EigenV objects
    Returns: overlap matrix overlap[m,n] = <evs0_m | evs1_n>
    """
    overlap = np.zeros((len(evs0), len(evs0)), dtype=complex)
    for m, ev0 in enumerate(evs0):
        for n, ev1 in enumerate(evs1):
            # below are old methods, they are quick, but assume each ev has the same gvecs in the same order

            # overlap[m, n] = np.vdot(ev0.get_evec_complex(),
            #                         ev1.get_evec_complex())
            # min_gvecs = min([len(ev0.get_evec_complex()), len(ev1.get_evec_complex())])
            # overlap[m, n] = np.vdot(ev0.get_evec_complex()[:min_gvecs],
            #                         ev1.get_evec_complex()[:min_gvecs])
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


def find_min_singular_value(wfc0, wfc1):
    """
    Args: wfc0, wfc1: each a list of Kpoint objects
    Returns: The (value, coordinate) of the smallest of all singular values
    across all kpoints from the singular value decomposition
    preformed at each kpoint, if this is too small the states
    were unable to be reasonably aligned
    """
    s_mins = np.zeros(len(wfc0))
    for i, kpt0, kpt1 in zip(count(), wfc0, wfc1):
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
                fromkpt = kpt_string[i]
                tokpt = kpt_string[0]
                overlap = compute_overlap(kpt_string[i].get_occupied_only(),
                                          kpt_string[0].get_occupied_only(),
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


def get_string(wfc, kx, ky):
    result = []
    for kpt in wfc:
        in_this_string = ((abs(kpt.kcoords[0] - kx) < 1.e-5)
                          and (abs(kpt.kcoords[1] - ky) < 1.e-5))
        if in_this_string:
            result.append(kpt)
    result.sort(key=lambda k: k.kcoords[-1])
    return result


def get_berry_phase_polarization(wfc, method=None):
    bz_2d_points = []
    for kpt in wfc:
        bz_2d_points.append((kpt.kcoords[0], kpt.kcoords[1]))

    nstr = len(set(bz_2d_points))

    if method is None or method == 'det_avg':
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

    else:
        bps = []
        for kx, ky in set(bz_2d_points):
            print()
            this_string = get_string(wfc, kx, ky)
            bp_s = bphase_along_string(this_string, cheap_pt=False)
            bp = bphase_with_mult(this_string, cheap_pt=False)
            bps.append(bp)
            print(kx, "\t", ky, '\t\t', (2 * bp) / (2*np.pi), (2 * bp_s) / (2*np.pi))
        polb = 2 * sum(bps) / (2 * np.pi * len(bps))
    return polb


def get_kpoint2_aligned_with_kpoint1(kpoint1, kpoint2):
    """returns a Kpoint object corresponding to kpoint2
    with a unitary operation applied such that bands
    are maximally aligned with kpoint1
    """
    raw_overlap = compute_overlap(kpoint1, kpoint2)
    u, s, v = np.linalg.svd(raw_overlap)
    rot_mat = np.linalg.inv(np.dot(u, v))
    new_eigenvs = np.zeros((len(kpoint2), len(kpoint2[0].evec), 2))
    for i in range(len(kpoint2)):
        for j in range(len(kpoint2)):
            # TODO: Should really just make EigenV store the complex number directly
            new_eigenvs[i, :, 0] += (rot_mat[i, j].real * kpoint2[j].evec[:, 0]
                                     - rot_mat[i, j].imag * kpoint2[j].evec[:, 1])
            new_eigenvs[i, :, 1] += (rot_mat[i, j].real * kpoint2[j].evec[:, 1]
                                     + rot_mat[i, j].imag * kpoint2[j].evec[:, 0])
    new_kpt = Kpoint(kpoint2.kcoords, kpoint2.weight,
                     kpoint2.planewaves, eigenvs=new_eigenvs)
    return new_kpt


# def compute_phase_diff_along_string(wfc0, wfc1, kx, ky):
#     tot_phase_change = 0.
#     for kpt0, kpt1 in zip(wfc0, wfc1):
#         print(kpt0.kcoords, kpt1.kcoords)
#         part_of_string = (kpt0.kcoords[0] == kx
#                           # and kpt0.kcoords[2] >= 0.
#                           and kpt0.kcoords[1] == ky)
#         if part_of_string:
#             overlap = compute_overlap(kpt0.get_occupied_only(),
#                                       kpt1.get_occupied_only())
#             u, s, v = np.linalg.svd(overlap)
#             unit_overlap = np.dot(u, v)
#             phase_change = 2 * log(np.linalg.det(unit_overlap)).imag / (2 * np.pi)
#             no_pt_phase_change = 2 * log(np.linalg.det(overlap)).imag / (2 * np.pi)
#             print(kpt0.kcoords, np.linalg.det(unit_overlap), " ", phase_change)
#             print(no_pt_phase_change)
#             tot_phase_change += phase_change
#     return tot_phase_change


if __name__ == '__main__':
    import sys

    print("reading {}".format(sys.argv[1]))
    wfc0 = read_wfc(sys.argv[1])

    # bp0 = get_berry_phase_polarization(wfc0, method='verbose')
    # bp0 = get_berry_phase_polarization(wfc0)
    # print("Electronic berry phase: {}".format(bp0))

    print("reading {}".format(sys.argv[2]))
    wfc1 = read_wfc(sys.argv[2])

    kpt_a0 = wfc0[0].get_occupied_only()
    kpt_b0 = wfc1[0].get_occupied_only()

    overlap_ab0 = compute_overlap(kpt_a0, kpt_b0)
    phase_ab0 = np.log(np.linalg.det(overlap_ab0)).imag
    print("phase a-b at gamma = {}".format(phase_ab0))
    u, s, v = np.linalg.svd(overlap_ab0)
    pt_overlap_ab0 = np.dot(u, v)
    pt_phase_ab0 = np.log(np.linalg.det(pt_overlap_ab0)).imag
    print("cheap_pt phase a-b at gamma = {}".format(pt_phase_ab0))

    kpt_b0_rot = get_kpoint2_aligned_with_kpoint1(kpt_a0, kpt_b0)
    overlap_true_pt = compute_overlap(kpt_a0, kpt_b0)
    phase_true_pt = np.log(np.linalg.det(overlap_true_pt)).imag
    print("true_pt phase a-b at gamma = {}".format(phase_true_pt))


    # # find kpt indices along 0,0
    # for i, k in enumerate(wfc0):
    #     if k.kcoords[0] == 0 and k.kcoords[1] == 0:
    #         print(i, k.kcoords)
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
