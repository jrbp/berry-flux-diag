#!/usr/bin/env python
from collections import MutableSequence
from itertools import count
from cmath import log
import numpy as np


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
        return self.evec[:, 0] + self.evec[:, 1]*1j


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
        if eigenvs:
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


def read_wfc(wfc_file, return_first_k=False):
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
            line_data = line.split()
            if len(line_data) != 7:
                raise ValueError('expected a eigenv header line')
            else:
                (planewaves, pws_in_calc, weight,
                 kx, ky, kz, occ) = [float(x) for x in line_data]
                planewaves = int(planewaves)
                try:
                    if this_kpoint.kcoords != (kx, ky, kz):
                        print("finished reading kpoint "
                              "{}".format(this_kpoint.kcoords))
                        if return_first_k:
                            return this_kpoint
                        kpoints.append(this_kpoint)
                        this_kpoint = Kpoint((kx, ky, kz), weight, planewaves)
                except NameError:
                    this_kpoint = Kpoint((kx, ky, kz), weight, planewaves)
                gvecs = np.zeros((planewaves, 3), dtype=float)
                evec = np.zeros((planewaves, 2), dtype=float)
                for i in range(planewaves):
                    (gvecs[i, 0], gvecs[i, 1], gvecs[i, 2],
                     evec[i, 0], evec[i, 1]) = f.readline().split()
                this_kpoint.append(EigenV(occ, gvecs, evec))
    print("finished reading kpoint {}".format(this_kpoint.kcoords))
    kpoints.append(this_kpoint)
    return kpoints


def compute_overlap(evs0, evs1):
    """
    Args: evs0, evs1: each a list of EigenV objects
    Returns: overlap matrix overlap[m,n] = <evs0_m | evs1_n>
    """
    overlap = np.zeros((len(evs0), len(evs0)), dtype=complex)
    for m, ev0 in enumerate(evs0):
        for n, ev1 in enumerate(evs1):
            overlap[m, n] = np.vdot(ev0.get_evec_complex(),
                                    ev1.get_evec_complex())
    return overlap


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


def compute_phase_diff_along_string(wfc0, wfc1, kx, ky):
    tot_phase_change = 0.
    for kpt0, kpt1 in zip(wfc0, wfc1):
        part_of_string = (kpt0.kcoords[0] == kx
                          and kpt0.kcoords[2] >= 0.
                          and kpt0.kcoords[1] == ky)
        if part_of_string:
            overlap = compute_overlap(kpt0.get_occupied_only(),
                                      kpt1.get_occupied_only())
            u, s, v = np.linalg.svd(overlap)
            unit_overlap = np.dot(u, v)
            phase_change = (-1 * log(np.linalg.det(unit_overlap)).imag)
            print(kpt0.kcoords, " ", phase_change)
            tot_phase_change += phase_change
    return tot_phase_change


if __name__ == '__main__':
    import sys

    print("reading {}".format(sys.argv[1]))
    wfc0 = read_wfc(sys.argv[1])
    print("reading {}".format(sys.argv[2]))
    wfc1 = read_wfc(sys.argv[2])

    print("smallest singular value is "
          "{} at the point {}".format(*find_min_singular_value(wfc0, wfc1)))

    bz_2d_points = []
    for kpt in wfc0:
        bz_2d_points.append((kpt.kcoords[0], kpt.kcoords[1]))

    string_vals = []
    num_strings = len(set(bz_2d_points))
    for kx, ky in set(bz_2d_points):
        print(kx, ", ", ky)
        val = compute_phase_diff_along_string(wfc0, wfc1, kx, ky)
        string_vals.append(val)
        print(val)
        print()
    print(sum(string_vals)/num_strings)
