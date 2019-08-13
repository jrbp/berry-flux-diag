#!/usr/bin/env python
# import os
import time
import logging
# import itertools
import json
# from multiprocessing import Pool
from collections.abc import MutableMapping
import numpy as np
from abipy.waves import WfkFile
from abipy.core.kpoints import Kpoint

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def compute_overlap(waves0, waves1, space='r'):
    """ returns overlap matrix between each list of PWWaveFunction """
    mesh0 = waves0[0].mesh
    overlap = np.zeros((len(waves0), len(waves1)), dtype=complex)
    if space == 'g':
        # put all on mesh once instead of using braket where it happens for each element
        ug0_mesh = np.array([wv.gsphere.tofftmesh(mesh0, wv.ug) for wv in waves0], dtype=complex)
        ug1_mesh = np.array([wv.gsphere.tofftmesh(mesh0, wv.ug) for wv in waves1], dtype=complex)
        for i, wv0 in enumerate(ug0_mesh): # possible to speedup with usage of np.tensordot?
            for j, wv1 in enumerate(ug1_mesh):
                overlap[i, j] = np.vdot(wv0, wv1)
    else:
        for i, wv0 in enumerate(waves0): # possible to speedup with usage of np.tensordot?
            for j, wv1 in enumerate(waves1):
                overlap[i, j] = wv0.braket(wv1, space=space)
    return overlap

def direction_to_vals(direction):
    if direction == 'z':
        comps = (0, 1)
        dir_comp = 2
        gvec = [0, 0, 1]
    elif direction == 'x':
        comps = (1, 2)
        dir_comp = 0
        gvec = [0, 1, 0]
    elif direction == 'y':
        comps = (0, 2)
        dir_comp = 1
        gvec = [1, 0, 0]
    else:
        raise ValueError('Direction must be x, y, or z')
    return comps, dir_comp, gvec

def get_strings(kpoints, direction):
    """ given a list of Kpoints and a direction x, y, z
        return a list of strings of kpoints along the corresponding direction """
    # includes the extra kpoint shifted by a gvector
    # adapted from some very old code, super inefficient, should clean up
    # but it's also pretty much never a bottleneck

    comps, dir_comp, gvec = direction_to_vals(direction)
    bz_2d_set = sorted(set([tuple((kpt.frac_coords[i] for i in comps))
                            for kpt in kpoints]))
    strings = []
    for bz_2d_pt in bz_2d_set:
        this_string = []
        for kpt in kpoints:
            in_this_string = ((abs(kpt.frac_coords[comps[0]] - bz_2d_pt[0]) < 1.e-5)
                              and (abs(kpt.frac_coords[comps[1]] - bz_2d_pt[1]) < 1.e-5))
            if in_this_string:
                this_string.append(kpt)
        this_string.sort(key=lambda k: k.frac_coords[dir_comp])
        this_string.append(this_string[0] + Kpoint(gvec, kpoints.reciprocal_lattice))
        strings.append(this_string)
    return strings


class Overlaps(MutableMapping):
    def __init__(self, wfk_files, rspace_translations=None):
        self.__dict__ = {} #TODO: possibly fill with keys of neighboring points with values None
        self._wfk_files = wfk_files #TODO: check that they have same Kpoints, maybe same lattice vectors
        if rspace_translations is not None:
            self._rspace_trans = np.array(rspace_translations)
        else:
            self._optimize_rspace_trans()

        # below option to accept paths doesn't close files
        # for wfk_file in wfk_files:
        #     if type(wfk_file) is str:
        #         self._wfk_files.append(WfkFile(wfk_file))
        #     elif type(wfk_file) is WfkFile:
        #         self._wfk_files = wfk_file
        #     else:
        #         raise TypeError('WFK files must be strings of nc file path or ')

    def __getitem__(self, key):
        if self.__dict__.get(key) is None:
            LOGGER.debug("{} not in Overlaps".format(key))
            if (key[1], key[0]) in self.__dict__:
                # if switched version present return hermitian conjugate
                LOGGER.debug("obtaining from Hermitian conj of {}, {}".format(key[1], key[0]))
                return self.__dict__[(key[1], key[0])].T.conj()
            else:
                self.__dict__[key] = self._compute_overlap(key)
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        self.__dict__.__delitem__(key)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    @property
    def kpoints(self):
        """ Return the Kpoints of the wfk files """
        return self._wfk_files[0].kpoints

    @property
    def rspace_trans(self):
        """ Return the set of real space translations on the wavefunctions """
        return self._rspace_trans

    @rspace_trans.setter
    def rspace_trans(self, new_trans):
        """ set real space translations on the wavefunctions
            overlaps involving the changed states will be deleted """
        old_trans = self._rspace_trans
        for i, (old, new) in enumerate(zip(old_trans, new_trans)):
            if (abs(old - new) > 1e-5).all():
                for key in self:
                    if i == key[0][0] or i == key[1][0]:
                        del self[key]
        self._rspace_trans = new_trans

    def _optimize_rspace_trans(self, polar_dir=None, min_sing_tol=0.1):
        # done this way to only read wfks from file once then keep operating on them
        # currently only optimized first wfk with respect to second
        # i.e. doesn't support more than two lambda states
        from scipy.optimize import brute

        if polar_dir is None:
            polar_dir = np.array([0., 0., 1.])

        LOGGER.info("finding real space translation along {}".format(polar_dir))

        # read in initial wavefunctions with no translation
        self._rspace_trans = np.array([[0., 0., 0.] for ll in self._wfk_files])
        wfks = [[self._get_pwws_at_state((i, kpt))
                 for kpt in self.kpoints]
                for i in range(2)]

        def min_sing_from_trans(trans):
            # start_time = time.time()
            # LOGGER.debug("trying trans {}".format(trans))
            wfk0_trans = [[pww.pww_rspace_translation(trans * polar_dir)
                           for pww in kpt] for kpt in wfks[0]]
            # read_time = time.time()
            # LOGGER.debug("\t translating wfks time {}".format(read_time - start_time))
            tmp_overlaps = [compute_overlap(wfk0, wfk1, space='gsphere')
                            for wfk0, wfk1 in zip(wfk0_trans, wfks[1])]
            # overlap_time = time.time()
            # LOGGER.debug("\t computed ovlps time {}".format(overlap_time - read_time))
            s_mins = []
            for ovl in tmp_overlaps:
                s = np.linalg.svd(ovl, compute_uv=False).min()
                s_mins.append(s)
                # LOGGER.debug(s)
            # svd_time = time.time()
            # LOGGER.debug("\t finished svd time {}".format(svd_time - overlap_time))
            return -1 * min(s_mins)
        minimize_res = brute(min_sing_from_trans, [[-0.2, 0.2]], Ns=6, full_output=True)
        LOGGER.debug(minimize_res)
        if minimize_res[1] < -1 * min_sing_tol:
            wfc0_trans = minimize_res[0][0] * np.array(polar_dir)
        else:
            raise Exception('Unable to align wavefunctions with a translation')
        LOGGER.info("Will translate first wfk by {}".format(wfc0_trans))
        self.rspace_trans = np.array([wfc0_trans] + [t for t in self.rspace_trans[1:]])

    def _get_pwws_at_state(self, state, spin=0):
        """ given l, kpt wher l is the lambda index
        and kpt is an abipy Kpoint object
        return a list of plane wave wavefunctions
        translated in rspace according to self._rspace_trans
        and shifted by g vectors as needed
        currently returns occupied bands only"""
        l, kpt = state
        wfk = self._wfk_files[l]
        occ_fact = 2 if wfk.nsppol == 1 else 1
        # LOGGER.debug("getting kpoint {}".format(kpt))
        pwws = [wfk.get_wave(spin, kpt, i)
                for i in range(wfk.nband)
                if wfk.ebands.occfacts[0][wfk.kindex(kpt)][i] == occ_fact]
        # LOGGER.debug("recieved kpoint {}".format(pwws[0].kpoint))

        if (abs(self.rspace_trans[l]) > 1e-5).any():
            pwws = [pww.pww_rspace_translation(self.rspace_trans[l]) for pww in pwws]

        diff = kpt - pwws[0].kpoint
        if (abs(diff.frac_coords) > 1e-5).any():
            if (np.abs(diff.frac_coords.round() - diff.frac_coords) < 1e-5).all():
                LOGGER.debug("translating {} by gvec {} to get {}".format(pwws[0].kpoint,
                                                                          diff.frac_coords.round(),
                                                                          kpt))
                # DO NOT TRANSLATE INPLACE, THEY ALL POINT TO SAME KPT
                # WILL BE SHIFTED ONCE FOR EACH BAND
                # maybe we really need an object that stores all pwws at a kpt
                # for now this works, but likely uses extra memory
                pwws = [pww.pww_translation(diff.frac_coords.round().astype(int)) for pww in pwws]

            else:
                raise ValueError("Could not get this wavefunction at kpt = {}".format(kpt))
        return pwws

    def _compute_overlap(self, pair_of_states):
        state0, state1 = pair_of_states
        if state0[1] == state1[1]:
            space = 'gsphere'
        else:
            space = 'r'
        return compute_overlap(self._get_pwws_at_state(state0),
                               self._get_pwws_at_state(state1),
                               space=space)

    def compute_string_overlaps(self, string):
        """ Compute and store all overlaps along the string (list of Kpoints)
            also computes all cross structure overlaps along string """
        # TODO: make cross structure overlaps optional? (they might already be there)
        wfk_file_indicies = range(len(self._wfk_files))
        string_indicies = range(len(string))
        all_wfcs_on_string = [[self._get_pwws_at_state((l, kpt)) for kpt in string]
                              for l in wfk_file_indicies]
        # DEBUGGING IndexError
        # first all cross structure overlaps
        for i, j in zip(wfk_file_indicies[:-1], wfk_file_indicies[1:]):
            for k in string_indicies:
                if ((i, string[k]), (j, string[k])) not in self.__dict__:
                    self[((i, string[k]), (j, string[k]))] = compute_overlap(all_wfcs_on_string[i][k],
                                                                             all_wfcs_on_string[j][k],
                                                                             space='gsphere')
        # now all cross BZ overlaps
        for i in wfk_file_indicies:
            for k0, k1 in zip(string_indicies[:-1], string_indicies[1:]):
                try:
                    self[((i, string[k0]), (i, string[k1]))] = compute_overlap(all_wfcs_on_string[i][k0],
                                                                               all_wfcs_on_string[i][k1],
                                                                               space='r')
                except IndexError:
                    LOGGER.info("wfk{}, {}->{}".format(i, string[k0], string[k1]))
                    LOGGER.info("ugs: {} and {}".format(all_wfcs_on_string[i][k0][0].ug.shape,
                                                        all_wfcs_on_string[i][k1][0].ug.shape))
                    raise IndexError
                LOGGER.debug("wfk{}, {}->{} finished".format(i, string[k0], string[k1]))

    def compute_all_string_overlaps(self, direction):
        """ Compute many overlaps at once
        so wavefunctions only need to be read and translated once
        direction is x, y, or z
        computes all overlaps between neighboring kpoints of strings along this direction
        also computes all cross lambda overlaps"""
        strings = get_strings(self.kpoints, direction)
        # TODO: possibly parallelize over strings
        for string in strings:
            self.compute_string_overlaps(string)

def find_min_singular_value_cross_structs(overlaps, s_vals_file=None):
    """ find the smallest singular value across all structures """
    s_mins = []
    if s_vals_file is not None:
        s_val_dict = {}
    for kpt in overlaps.kpoints:
        this_u = overlaps[((0, kpt), (1, kpt))]
        s = np.linalg.svd(this_u, compute_uv=False)
        if s_vals_file is not None:
            s_val_dict[str(kpt)] = list(s)
        s_mins.append(s.min())
    if s_vals_file is not None:
        with open(s_vals_file, 'w') as f:
            json.dump(s_val_dict, f, indent=3)
    s_mins = np.array(s_mins)
    return s_mins.min(), overlaps.kpoints[s_mins.argmin()]

if __name__ == '__main__':
    from argparse import ArgumentParser
    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument("wfc_files", nargs=2,
                            help="netcdf wavefunction files for each structure")
    ARG_PARSER.add_argument("-l", "--log_file", required=False,
                            default="pt_phase.log", help="log file name, default is pt_phase.log")
    ARG_PARSER.add_argument("-n", "--num_cpus", required=False, default=4, type=int,
                            help="number of cpus to parallelize over")
    ARG_PARSER.add_argument("-sf", "--singular_values_file", required=False,
                            help="file name to write singular values (as json)")
    ARG_PARSER.add_argument("-d", "--direction", required=False,
                            help="direction to compute change in polarization", default='z')
    ARG_PARSER.add_argument("-t", "--translation", required=False, default=None, type=float,
                            help="translation to use on wavefunction")
    ARG_PARSER.add_argument("-s", "--min_s_tol", required=False, default=0.1, type=float,
                            help="minimum singular value accepted in alignment, default is 0.1")
    ARGS = ARG_PARSER.parse_args()

    STREAM_HANDLER = logging.StreamHandler()
    STREAM_HANDLER.setLevel(logging.INFO)
    LOGGER.addHandler(STREAM_HANDLER)

    FILE_HANDLER = logging.FileHandler(filename=ARGS.log_file, mode='w')
    FILE_HANDLER.setLevel(logging.DEBUG)
    LOGGER.addHandler(FILE_HANDLER)

    wfc0 = WfkFile(ARGS.wfc_files[0])
    wfc1 = WfkFile(ARGS.wfc_files[1])

    LOGGER.info("Structure 0")
    LOGGER.info(wfc0.structure)
    LOGGER.info("Structure 1")
    LOGGER.info(wfc1.structure)

    # we will translate wfc0 to maximize the smallest singular value
    # not done until wfc is actually needed
    if ARGS.translation:
        # TODO: should read direction arg and set appropriatly
        rspace_trans = ARGS.translation * np.array([[0., 0., 1.], [0., 0., 0.]])
    else:
        # LOGGER.info("Finding translation to align wavefunctions")
        # wfc0_rspace_trans = find_translation_to_align_w1_with_w2(wfc0, wfc1, min_sing_tol=ARGS.min_s_tol)
        rspace_trans = None
    overlaps = Overlaps((wfc0, wfc1), rspace_trans)

    if ARGS.singular_values_file:
        min_s = find_min_singular_value_cross_structs(overlaps, s_vals_file=ARGS.singular_values_file)

    start_time = time.time()
    overlaps.compute_all_string_overlaps(ARGS.direction)
    after_overlap_time = time.time()
    LOGGER.debug("time info: {} seconds to compute all overlaps".format(after_overlap_time - start_time))
    strings = get_strings(wfc0.kpoints, ARGS.direction)
    string_phases = []
    for string in strings:
        inner_loop_sum = 0.
        for kpt0, kpt1 in zip(string[:-1], string[1:]):
            loops = [((0, kpt0), (1, kpt0)),
                     ((1, kpt0), (1, kpt1)),
                     ((1, kpt1), (0, kpt1)),
                     ((0, kpt1), (0, kpt0)),]
            curly_U = np.identity(len(overlaps[loops[0]]))
            for states in loops:
                M = overlaps[states]
                u, s, v = np.linalg.svd(M)
                smallest_sing_val = min(s)
                LOGGER.debug(("finding curly M: \n"
                              " state1: {} \n"
                              " state2: {} \n"
                              " min singular value: {}").format(
                                  states[0], states[1], smallest_sing_val))
                if smallest_sing_val < 0.1:
                    LOGGER.warning("MIN SINGULAR VALUE OF {} FOUND!".format(
                        smallest_sing_val))
                curly_M = np.dot(u, v)
                #LOGGER.debug('\n')
                #LOGGER.debug(M)
                #LOGGER.debug('\n')
                #LOGGER.debug(curly_M)
                curly_U = np.dot(curly_U, curly_M)
            wlevs = np.log(np.linalg.eigvals(curly_U)).imag
            inner_loop_sum += sum(wlevs) / np.pi
        string_phases.append(inner_loop_sum)
    string_sum = sum(string_phases)
    LOGGER.debug("time info: {} seconds".format(time.time() - after_overlap_time))

    for string, val in zip(strings, string_phases):
        LOGGER.info("{}, {}: {}".format(string[0].frac_coords[0], string[0].frac_coords[1], val))
    LOGGER.info("average across strings: {}".format(string_sum / len(strings)))
    wfc0.close()
    wfc1.close()
