#!/usr/bin/env python
from collections import MutableSequence
import logging
import itertools
# from itertools import count
# from cmath import log
import numpy as np
from scipy.optimize import brute
from numba import jit
from multiprocessing import Pool

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    def real_space_trans(self, trans):
        new_eigenvs = []
        for ev in self._eigenvs:
            f1 = np.e**(-2 * np.pi * 1.j * np.dot(self.kcoords, np.array(trans)))
            f2 = np.e**(-2 * np.pi * 1.j * np.inner(ev.gvecs, np.array(trans)))
            new_amps = ev.get_evec_complex() * f2  * f1
            new_eigenvs.append(EigenV(ev.occupation, ev.gvecs,
                                      np.stack((new_amps.real, new_amps.imag), axis=1)))
        return Kpoint(self.kcoords, self.weight, self.planewaves, new_eigenvs)


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
            # planewaves = int(max_planewaves)  # if fft used
            planewaves = int(pws_in_calc)   # if fft not used
            try:
                if (np.abs(np.array(this_kpoint.kcoords) - np.array([kx, ky, kz])) > 1.e-5).any():
                    logger.debug("finished reading kpoint "
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
    logger.debug("finished reading kpoint {}".format(this_kpoint.kcoords))
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
    try:
        kc0 = evs0.kcoords
        kc1 = evs1.kcoords
        if np.all(kc0 == kc1):
            overlap = compute_overlap_same_gvecs(evs0, evs1)
            return overlap
    except Exception as e:
        logger.warning('{} in compute overlap with same gvecs')
        pass
    evs0_arr = np.array([ev0.get_evec_complex() for ev0 in evs0])
    evs1_arr = np.array([ev1.get_evec_complex() for ev1 in evs1])
    evs0_gs = np.array([ev0.gvecs for ev0 in evs0])
    evs1_gs = np.array([ev1.gvecs for ev1 in evs1])
    if len(evs0) != len(evs1):
        logger.warning('different number of eigenvecs between kpoints {} and {}'.format(kc0, kc1))
    overlap = np.zeros((len(evs0), len(evs0)), dtype=complex)
    overlap = compute_overlap_jit(overlap, evs0_arr, evs1_arr, evs0_gs, evs1_gs, dg)
    return overlap
    # overlap = np.zeros((len(evs0), len(evs0)), dtype=complex)
    # for m, ev0 in enumerate(evs0):
    #     for n, ev1 in enumerate(evs1):
    #         # # slow non jit way
    #         # this_element = 0.
    #         # for gvec0, ipw0 in zip(ev0.gvecs, np.conj(ev0.get_evec_complex())):
    #         #     for gvec1, ipw1 in zip(ev1.gvecs, ev1.get_evec_complex()):
    #         #         if all(gvec0 == gvec1):
    #         #             this_element += ipw0 * ipw1
    #         overlap[m, n] = compute_overlap_element(ev0.gvecs, np.conj(ev0.get_evec_complex()),
    #                                                 ev1.gvecs, ev1.get_evec_complex(), dg)
    # return overlap


@jit(nopython=True, cache=True)
def compute_overlap_jit(overlap, evs0, evs1, evs0_gs, evs1_gs, dg=np.array([0., 0., 0.])):
    for m in range(len(evs0)):
        for n in range(len(evs1)):
            overlap[m, n] = compute_overlap_element(evs0_gs[m], np.conj(evs0[m]),
                                                    evs1_gs[n], evs1[n],
                                                    dg)
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
    for i, kpt0, kpt1 in zip(itertools.count(), wfc0, wfc1):
        if same_gvecs:
            overlap = compute_overlap_same_gvecs(kpt0.get_occupied_only(),
                                                 kpt1.get_occupied_only())
        else:
            overlap = compute_overlap(kpt0.get_occupied_only(),
                                      kpt1.get_occupied_only())
        s = np.linalg.svd(overlap, compute_uv=False)
        s_mins[i] = s.min()
    return s_mins.min(), wfc0[s_mins.argmin()].kcoords


def translation_to_align_w1_with_w2(wfc0, wfc1, polar_dir=[0, 0, 1]):
    def min_sing_from_trans(trans):
        translated_wfc = [kpt.real_space_trans(trans * np.array(polar_dir)) for kpt in wfc0]
        return -1 * find_min_singular_value(translated_wfc, wfc1, same_gvecs=True)[0]
    minimize_res = brute(min_sing_from_trans, [[-0.5, 0.5]], Ns=30, full_output=True)
    logger.debug(minimize_res)
    if minimize_res[1] < -0.1:
        return minimize_res[0][0] * np.array(polar_dir)
    else:
        raise Exception('Unable to align wavefunctions with a translation')


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
    logger.debug("aligning: \n kpoint1: {} \n kpoint2: {} \n min singular value: {}".format(
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


def strings_to_loops(string1, string2): # now adds last pt to close loop
    # IMPORTANT: currently hard coded for strings along kz
    # may want to add check that string1 and string2 are same length
    num_pts_in_string = len(string1)
    loops = []
    for i in range(num_pts_in_string):
        if i+1 == num_pts_in_string:
            loops.append([string1[i], string2[i],
                          string2[0].get_g_shifted([0, 0, 1]),
                          string1[0].get_g_shifted([0, 0, 1]), string1[i]])
        else:
            loops.append([string1[i], string2[i], string2[i+1], string1[i+1], string1[i]])
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


def get_overlaps_along_path(kpt_path):     # NO LONGER SHIFTS LAST PT
    nkpts = len(kpt_path)
    overlaps = []
    pairs_of_ks = []
    for i in range(nkpts - 1):
        this_kpt = kpt_path[i].get_occupied_only()
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
    logger.debug("min sing val from indiv bphase step = {}".format(min(s)))
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
        logger.debug("{} {}".format(kx, ky))
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


def curly_u_from_path(kpt_path):
    overlaps, kpt_pairs = get_overlaps_along_path(kpt_path)
    curly_U = np.identity(len(overlaps[0][0]))
    for M, kpt_p in zip(overlaps, kpt_pairs):
        u, s, v = np.linalg.svd(M)
        smallest_sing_val = min(s)
        logger.debug(("finding curly M: \n"
                      " kpoint1: {} \n"
                      " kpoint2: {} \n"
                      " min singular value: {}").format(
                          kpt_p[0], kpt_p[1], smallest_sing_val))
        if smallest_sing_val < 0.1:
            logger.warning('MIN SINGULAR VALUE OF {} FOUND!'.format(
                smallest_sing_val))
        curly_M = np.dot(u, v)
        #logger.debug('\n')
        #logger.debug(M)
        #logger.debug('\n')
        #logger.debug(curly_M)
        curly_U = np.dot(curly_U, curly_M)
    return curly_U

def pt_phase_from_path(kpt_path):
    curly_U = curly_u_from_path(kpt_path)
    wlevs = np.log(np.linalg.eigvals(curly_U)).imag
    logger.debug('loop eigenvalues:\n {}'.format(wlevs))
    return sum(wlevs)


# fro spont pol, should change name of function
def pt_phase_from_strings(bz_2d_pt, wfc0, wfc1):
    kx, ky = bz_2d_pt
    logger.info("strings along {}, {}:".format(kx, ky))
    loops = strings_to_loops(get_string(wfc0, kx, ky),
                             get_string(wfc1, kx, ky))
    inner_loop_sum = 0.
    for loop in loops:
        inner_loop_sum += pt_phase_from_path(loop) / np.pi
    # loop_pool = Pool(4)
    # inner_loop_vals = loop_pool.map(pt_phase_from_path, loops)
    # loop_pool.close()
    # loop_pool.join()
    # inner_loop_sum = sum(inner_loop_vals) / np.pi
    logger.debug('this strings change in phase: {}'.format(inner_loop_sum))
    return inner_loop_sum


if __name__ == '__main__':
    import sys
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    if len(sys.argv) > 3:
        logfile = sys.argv[3]
    else:
        logfile = 'pt_phase.log'
    file_handler = logging.FileHandler(filename=logfile, mode='w')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info("reading {}".format(sys.argv[1]))
    wfc0 = read_wfc(sys.argv[1])

    logger.info("reading {}".format(sys.argv[2]))
    wfc1 = read_wfc(sys.argv[2])

    # translate wfc0 to maximize the smallest singular value
    logger.info("Finding translation to align wavefunctions")
    trans = translation_to_align_w1_with_w2(wfc0, wfc1)
    logger.info("Translating {} in real space by {}".format(sys.argv[1], trans))
    wfc0 = [kpt.real_space_trans(trans) for kpt in wfc0]

    # saving time on the PTO sc cell version, since I've already found the translation
    # wfc0 = [kpt.real_space_trans([ 0., 0., 0.04154095]) for kpt in wfc0]

    bz_2d_set = sorted(set([(kpt.kcoords[0], kpt.kcoords[1]) for kpt in wfc0]))

    string_vals = []
    import time
    start_time = time.time()
    # for kx, ky in bz_2d_set:
    #     inner_loop_sum = pt_phase_from_strings((kx, ky), wfc0, wfc1)
    #     string_vals.append(inner_loop_sum)
    string_pool = Pool(4)
    string_vals = string_pool.starmap(pt_phase_from_strings,
                                      zip(bz_2d_set,
                                          itertools.repeat(wfc0),
                                          itertools.repeat(wfc1)))
    string_pool.close()
    string_pool.join()
    string_sum = sum(string_vals)
    logger.debug("time info: {} seconds".format(time.time() - start_time))
    logger.info("summary")
    for k, val in zip(bz_2d_set, string_vals):
        logger.info("{}: {}".format(k, val))
    logger.info("average across strings: {}".format(string_sum / len(bz_2d_set)))
