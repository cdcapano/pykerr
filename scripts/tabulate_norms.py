#! /usr/bin/env python

"""Tabulates normalization constants for the spins and modes saved in the
data hdf files, and saves the constants to them.

pykerr needs to be installed first, with the data files downloaded.
"""

import argparse
import h5py
import numpy
import multiprocessing
# override the MAX_SPIN so we cqn compute all values
import pykerr.qnm
pykerr.qnm.MAX_SPIN = 0.9999
from pykerr.harmonics import slmnorm

parser = argparse.ArgumentParser()
parser.add_argument('--input-file',
                    help='The hdf file. The file will be modified in place.')
parser.add_argument('-s', type=int, choices=[0, 1, 2],
                    help='The (-) spin weight. Must be either 0, 1, or 2.')
parser.add_argument('--npoints', type=int, default=1000,
                    help="The number of points to use in the integrals.")
parser.add_argument("--max-recursion", type=int, default=1000,
                    help="Maximun recursion to use for Slm calculation.")
parser.add_argument("--tol", type=float, default=1e-8,
                    help="The tolerance for the Slm calculation.")
opts = parser.parse_args()

# load the spins and modes from the data file
print("loading spins")
spins = {}
with h5py.File(opts.input_file, 'r') as fp:
    for mode in fp:
        spins[mode] = fp[mode]['spin'][()]

norms = {}
print("computing")
for mode in spins:
    print(mode)
    l, m, n = mode
    l, m, n = tuple(map(int, [l, m, n]))
    norm = numpy.zeros(spins[mode].shape)
    for ii, a in enumerate(spins[mode]):
        if not ii % 10:
            print("{} / {}".format(ii, norm.size), end="\r")
        norm[ii] = slmnorm(a, l, m, n, s=-opts.s, npoints=opts.npoints,
                                 tol=opts.tol,
                                 max_recursion=opts.max_recursion)
    print("")
    norms[mode] = numpy.array(list(norm))

# write
print("writing")
tmplt = '{}/' + 's{}'.format(opts.s) + 'norm'
with h5py.File(opts.input_file, 'a') as fp:
    for mode, norm in norms.items():
        group = tmplt.format(mode)
        if group not in fp:
            fp.create_dataset(group, norm.shape, dtype=numpy.float32,
                              compression="gzip")
        fp[group][:] = norm
print("done")
