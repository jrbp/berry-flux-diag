#!/usr/bin/env python
from collections import MutableSequence
# from itertools import count
# from cmath import log
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
        return Kpoint(tuple(np.array(self.kcoords) + np.array(g_shift)),
                      self.weight, self.planewaves,
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
            #             this is due to how the modified cut3d is writing wfcs
            #             this should be changed there
            planewaves = int(max_planewaves)  # if fft used
            # planewaves = int(pws_in_calc)   # if fft not used
            try:
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


def get_string(wfc, kx, ky):
    result = []
    for kpt in wfc:
        in_this_string = ((abs(kpt.kcoords[0] - kx) < 1.e-5)
                          and (abs(kpt.kcoords[1] - ky) < 1.e-5))
        if in_this_string:
            result.append(kpt)
    result.sort(key=lambda k: k.kcoords[-1])
    return result


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


def strings_to_single_loop(string1, string2):
    loop = [string1[0]] + string2
    loop.append(string2[0].get_g_shifted([0, 0, 1]))
    loop.append(string1[0].get_g_shifted([0, 0, 1]))
    loop += string1[:0:-1]
    return loop


def strings_to_loops(string1, string2):
    # IMPORTANT: currently hard coded for strings along kz
    # may want to add check that string1 and string2 are same length
    num_pts_in_string = len(string1)
    loops = []
    for i in range(num_pts_in_string):
        if i+1 == num_pts_in_string:
            loops.append([string1[i], string2[i],
                          string2[0].get_g_shifted([0, 0, 1]),
                          string1[0].get_g_shifted([0, 0, 1])])
        else:
            loops.append([string1[i], string2[i], string2[i+1], string1[i+1]])
    return loops


def align_bands_along_path(kpt_path):
    # this will likely be replaced by a method which simply uses overlaps
    # instead of constructing full pt gauge
    # so not being too careful about copying kpts or being efficient
    smooth_path = [{}] * len(kpt_path)
    first_point = True
    for i in range(len(kpt_path)):
        this_kpt = kpt_path[i].get_occupied_only()
        if first_point:
            smooth_path[i] = this_kpt
            last_point_ind = i
            first_point = False
        else:
            this_kpt_aligned = get_kpoint2_aligned_with_kpoint1(
                smooth_path[last_point_ind], this_kpt)
            smooth_path[i] = this_kpt_aligned
            last_point_ind = i
    return smooth_path


def get_overlaps_along_path(kpt_path):
    nkpts = len(kpt_path)
    overlaps = []
    pairs_of_ks = []
    for i in range(nkpts):
        this_kpt = kpt_path[i].get_occupied_only()
        if i == nkpts - 1:
            next_kpt = kpt_path[0].get_occupied_only()
        else:
            next_kpt = kpt_path[i+1].get_occupied_only()
        pairs_of_ks.append((this_kpt.kcoords, next_kpt.kcoords))
        raw_overlap = compute_overlap(this_kpt, next_kpt)
        overlaps.append(raw_overlap)
    return overlaps, pairs_of_ks


def get_indiv_band_bphases(kpoint1, kpoint2, returnEigVecs=False):
    raw_overlap = compute_overlap(kpoint1, kpoint2)
    u, s, v = np.linalg.svd(raw_overlap)
    # rot_mat = np.linalg.inv(np.dot(u, v))
    rot_mat = np.dot(u, v)
    rot_mat_eigenvals = np.linalg.eigvals(rot_mat)
    print("min sing val from indiv bphase step = {}".format(min(s)))
    return np.log(rot_mat_eigenvals).imag


def get_wfc2_aligned_with_wfc1(wfc1, wfc2):
    """returns a list of Kpoints object corresponding to wfc2
    with a unitary operation applied to each Kpoint such that bands
    are maximally aligned with corresponding Kpoints in wfc1
    kpoints must be in the same order"""
    return [get_kpoint2_aligned_with_kpoint1(kp1.get_occupied_only(),
                                             kp2.get_occupied_only())
            for kp1, kp2 in zip(wfc1, wfc2)]


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


def get_string_indicies(wfc, kx, ky):
    result = []
    for i, kpt in enumerate(wfc):
        in_this_string = ((abs(kpt.kcoords[0] - kx) < 1.e-5)
                          and (abs(kpt.kcoords[1] - ky) < 1.e-5))
        if in_this_string:
            result.append((i, kpt))
    result.sort(key=lambda k: k[1].kcoords[-1])
    return [entry[0] for entry in result]


if __name__ == '__main__':
    import sys

    print("reading {}".format(sys.argv[1]))
    wfc0 = read_wfc(sys.argv[1])

    # try localizing wfc0 first
    # wfc_occ0 = [k.get_occupied_only() for k in wfc0]
    # wfc_occ0_smooth = construct_pt_gauge_along_kz(wfc_occ0)
    # wfc0 = wfc_occ0_smooth
    # end localizing wfc0

    print("reading {}".format(sys.argv[2]))
    wfc1 = read_wfc(sys.argv[2])
    # wfc1 = get_wfc2_aligned_with_wfc1(wfc0, read_wfc(sys.argv[2]))

    bz_2d_set = sorted(set([(kpt.kcoords[0], kpt.kcoords[1]) for kpt in wfc0]))
    # print(bz_2d_set)

    # begin test of all small loops
    # string_sum = 0.
    # string_vals = []
    # for kx, ky in bz_2d_set:
    #     print("strings along {}, {}:".format(kx, ky))
    #     loops = strings_to_loops(get_string(wfc0, kx, ky),
    #                              get_string(wfc1, kx, ky))
    #     inner_loop_sum = 0.
    #     for loop in loops:
    #         pt_loop = align_bands_along_path(loop)
    #         wlevs_loop = get_indiv_band_bphases(pt_loop[-1], pt_loop[0])
    #         print(wlevs_loop)
    #         print()
    #         inner_loop_sum += sum(wlevs_loop) / np.pi
    #     print(inner_loop_sum)
    #     string_vals.append(inner_loop_sum)
    #     string_sum += inner_loop_sum
    # print()
    # print("summary")
    # for k, val in zip(bz_2d_set, string_vals):
    #     print("{}: {}".format(k, val))
    # print("average across strings:")
    # print(string_sum / len(bz_2d_set))
    # end test of all small loops

    # begin test of all small loops, but using eqn 3.119 instead of constructing pt gauge explicitly
    string_sum = 0.
    string_vals = []
    for kx, ky in bz_2d_set:
        print("strings along {}, {}:".format(kx, ky))
        loops = strings_to_loops(get_string(wfc0, kx, ky),
                                 get_string(wfc1, kx, ky))
        inner_loop_sum = 0.
        for loop in loops:
            overlaps, kpt_pairs = get_overlaps_along_path(loop)
            curly_U = np.identity(len(overlaps[0][0]))
            for M, kpt_p in zip(overlaps, kpt_pairs):
                u, s, v = np.linalg.svd(M)
                print("finding curly M: \n",
                      " kpoint1: {} \n",
                      " kpoint2: {} \n",
                      " min singular value: {}".format(
                          kpt_p[0], kpt_p[1], min(s)))
                curly_M = np.dot(u, v)
                curly_U = np.dot(curly_U, curly_M)
            wlevs_loop = np.log(np.linalg.eigvals(curly_U)).imag
            print(wlevs_loop)
            print()
            inner_loop_sum += sum(wlevs_loop) / np.pi
        print(inner_loop_sum)
        string_vals.append(inner_loop_sum)
        string_sum += inner_loop_sum
    print()
    print("summary")
    for k, val in zip(bz_2d_set, string_vals):
        print("{}: {}".format(k, val))
    print("average across strings:")
    print(string_sum / len(bz_2d_set))
    # end test of all small loops

    # begin test of all large loops (across all of kz)
    # string_vals = []
    # loop_sums = 0.
    # for kx, ky in bz_2d_set:
    #     print("strings along {}, {}:".format(kx, ky))
    #     loop = strings_to_single_loop(get_string(wfc0, kx, ky),
    #                                   get_string(wfc1, kx, ky))
    #     pt_loop = align_bands_along_path(loop)
    #     wlevs_loop = get_indiv_band_bphases(pt_loop[-1], pt_loop[0])
    #     print(wlevs_loop)
    #     print()
    #     loop_sums += sum(wlevs_loop) / np.pi
    #     string_vals.append(sum(wlevs_loop) / np.pi)
    # for k, val in zip(bz_2d_set, string_vals):
    #     print("{}: {}".format(k, val))
    # print("average across strings:")
    # print(loop_sums / len(bz_2d_set))
    # end test of all large loops (across all of kz)

    # begin test of one little loop
    # wfc0_strings = [get_string(wfc0, *k_2d) for k_2d in bz_2d_set]
    # wfc1_strings = [get_string(wfc1, *k_2d) for k_2d in bz_2d_set]
    # loops0 = strings_to_loops(wfc0_strings[0], wfc1_strings[0])
    # print([[kpt.kcoords for kpt in loop] for loop in loops0])
    # loop0 = loops0[0]
    # end test of one little loop

    # begin test of one large loop (across all of kz)
    # loop0 = strings_to_single_loop(wfc0_strings[4], wfc1_strings[4])
    # for kpt in loop0:
    #     print(kpt.kcoords)
    # print()
    # pt_loop0 = align_bands_along_path(loop0)
    # wlevs_loop0 = get_indiv_band_bphases(pt_loop0[-1], pt_loop0[0])
    # print(wlevs_loop0)
    # print(sum(wlevs_loop0) / np.pi)
    # end test of one large loop (across all of kz)
