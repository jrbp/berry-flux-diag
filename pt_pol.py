#!/usr/bin/env python
from collections import MutableSequence
import numpy as np


class EigenV():
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


class Kpoint(MutableSequence):
    def __init__(self, kcoords, weight, planewaves, eigenvs=[]):
        self._kcoords = kcoords
        self._weight = weight
        self._planewaves = planewaves
        self._eigenvs = eigenvs

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

    def __setitem__(self, i, mode):
        if isinstance(mode, EigenV):
            self._eigenvs[i] = mode
        else:
            raise TypeError("Elements of Kpoint must be EigenV objects")


def read_wfc(wfc_file):
    kpoints = []
    with open(wfc_file) as f:
        for line in f:
            line_data = line.split()
            if len(line_data) != 7:
                raise ValueError('expected a eigenv header line')
            else:
                planewaves, weight, kx, ky, kz, en, occ = [float(x) for x in line_data]
                planewaves = int(planewaves)
                try:
                    if this_kpoint.kcoords != (kx, ky, kz):
                        print("read kpoint {}".format(this_kpoint.kcoords))
                        kpoints.append(this_kpoint)
                        this_kpoint = Kpoint((kx, ky, kz), weight, planewaves)
                except NameError:
                    this_kpoint = Kpoint((kx, ky, kz), weight, planewaves)
                gvecs = np.zeros((planewaves, 3), dtype=float)
                evec = np.zeros((planewaves, 2), dtype=float)
                for i in range(planewaves):
                    gvecs[i, 0], gvecs[i, 1], gvecs[i, 2], evec[i, 0], evec[i, 1] = f.readline().split()
                this_kpoint.append(EigenV(occ, gvecs, evec))
    return kpoints


if __name__ == '__main__':
    import sys
    wfc_file = sys.argv[1]
    wfc = read_wfc(wfc_file)
    print(wfc[0].kcoords)
    print(wfc[0][21].occupation)
    print(wfc[0][0].gvecs[0])
    print(wfc[0][0].evec)
