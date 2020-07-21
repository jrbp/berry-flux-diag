#!/usr/bin/env python3

"""This file should be a temporary measure.
  There are certain changes to abipy needed by the code not yet present in abipy.
  Instead of insisting that a fork of abipy must be used certain class methods are patched here.
  Hopefully these changes make their way upstream and this file can be removed."""

import numpy as np
from abipy.core.kpoints import Kpoint
from abipy.waves.pwwave import PWWaveFunction


def braket(self, other, space="g"):
    """
    Returns the scalar product <u1|u2> of the periodic part of two wavefunctions
    computed in G-space or r-space, depending on the value of space.
    Note that selection rules introduced by k-points is not taken into accout.

    Args:
        other: Other wave (right-hand side)
        space:  Integration space. Possible values ["g", "gsphere", "r"]
            if "g" or "r" the scalar product is computed in G- or R-space on the FFT box.
            if "gsphere" the integration is done on the G-sphere. Note that
            this option assumes that self and other have the same list of G-vectors.
    """
    space = space.lower()

    if space == "g":
        ug1_mesh = self.gsphere.tofftmesh(self.mesh, self.ug)
        ug2_mesh = other.gsphere.tofftmesh(self.mesh, other.ug) if other is not self else ug1_mesh
        return np.vdot(ug1_mesh, ug2_mesh)
    elif space == "gsphere":
        return np.vdot(self.ug, other.ug)
    elif space == "r":
        # relevant change: ensure both are on same mesh
        #return np.vdot(self.ur, other.ur) / self.mesh.size
        return np.vdot(self.ur, other.get_ur_mesh(self.mesh, copy=False)) / self.mesh.size
    else:
        raise ValueError("Wrong space: %s" % str(space))


def pww_translation(self, gvector, rprimd=None):
    """Returns the pwwave of the kpoint translated by one gvector."""
    gsph = self.gsphere.copy()
    wpww = PWWaveFunction(self.structure, self.nspinor, self.spin, self.band, gsph, self.ug.copy())
    # wpww.mesh = self.mesh
    wpww.set_mesh(self.mesh)
    wpww.pww_translation_inplace(gvector, rprimd)
    return wpww


def pww_translation_inplace(self, gvector, rprimd=None):
    """Translates the pwwave from 1 kpoint by one gvector."""
    if rprimd is None:
        rprimd = self.structure.lattice.matrix
    # self.gsphere.kpoint = self.gsphere.kpoint + gvector
    self.gsphere.kpoint = self.gsphere.kpoint + Kpoint(gvector, self.gsphere.kpoint.lattice)
    # self.gsphere.gvecs = self.gsphere.gvecs + gvector
    self.gsphere._gvecs = self.gsphere.gvecs - gvector
    # fft_ndivs = (self.mesh.shape[0] + 2, self.mesh.shape[1] + 2, self.mesh.shape[2] + 2)
    # newmesh = Mesh3D(fft_ndivs, rprimd, pbc=True)
    # self.mesh = newmesh
    self.delete_ur() # ur will get recomputed correctly as needed


def pww_rspace_translation(self, tau):
    gsph = self.gsphere.copy()
    wpww = PWWaveFunction(self.structure, self.nspinor, self.spin, self.band, gsph, self.ug.copy())
    wpww.set_mesh(self.mesh)
    wpww.pww_rspace_translation_inplace(tau)
    return wpww


def pww_rspace_translation_inplace(self, tau):
    f1 = np.e**(-2 * np.pi * 1.j * np.dot(self.kpoint.frac_coords, np.array(tau)))
    f2 = np.e**(-2 * np.pi * 1.j * np.dot(self.gsphere.gvecs, np.array(tau)))
    self.set_ug(self.ug * f2 * f1)


def PATCH_PWWaveFunction():
    # method overrides
    PWWaveFunction.braket = braket
    PWWaveFunction.pww_translation_inplace = pww_translation_inplace
    PWWaveFunction.pww_translation = pww_translation
    # new methods
    PWWaveFunction.pww_rspace_translation_inplace = pww_rspace_translation_inplace
    PWWaveFunction.pww_rspace_translation = pww_rspace_translation
