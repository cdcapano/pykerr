#! /usr/bin/env python

"""Tabulates normalization constants for the spins and modes saved in the
data hdf files, and saves the constants to them.

pykerr needs to be installed first, with the data files downloaded.
"""

import argparse
import h5py
import numpy
from multiprocessing import Pool
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
parser.add_argument("--tol", type=float, default=1e-8,
                    help="The tolerance for the Slm calculation.")
parser.add_argument("--maxtol", type=float, default=1e-4,
                    help="The max tolerance for the Slm calculation.")
parser.add_argument("--max-recursion", type=int, default=1000,
                    help="Maximun recursion to use for Slm calculation.")
parser.add_argument('--skip-if-exists', action='store_true', default=False,
                    help="Skip any modes that already have a norm written.")
parser.add_argument('--nprocesses', type=int, default=1,
                    help="Parallelize over the given number of processes. "
                         "Default is 1.")
opts = parser.parse_args()

# group name for norm
tmplt = '{}/' + 's{}norm'.format(opts.s)
nmtmplt = '{}/' + 's{}nmnorm'.format(opts.s)

# load the spins and modes from the data file
with h5py.File(opts.input_file, 'r') as fp:
    spins = fp['spin'][()]
    # get the modes to analyze
    modes = []
    print('getting modes')
    for name in fp:
        if name == 'spin':
            continue
        if opts.skip_if_exists:
            skip = tmplt.format(name) in fp
            if name[1] != '0':
                skip &= nmtmplt.format(name) in fp
            if skip:
                print("skipping {}".format(name))
                continue
        modes.append(name)
        
def getnorm(almn):
    a, l, m, n = almn
    return slmnorm(1e-4*a, l, m, n, s=-opts.s,
                   npoints=opts.npoints,
                   tol=opts.tol, maxtol=opts.maxtol,
                   max_recursion=opts.max_recursion,
                   use_cache=False)

if opts.nprocesses > 1:
    pool = Pool(opts.nprocesses)
    mfunc = pool.map
else:
    mfunc = map

print("computing")
for mode in modes:
    print(mode)
    l, m, n = mode
    l, m, n = tuple(map(int, [l, m, n]))
    # +m
    norm = numpy.array(list(mfunc(getnorm, [(a, l, m, n) for a in spins])))
    # -m
    if m != 0:
        nmnorm = numpy.array(list(mfunc(getnorm,
                                        [(a, l, -m, n) for a in spins])))
    # write
    print("writing", end="\r")
    with h5py.File(opts.input_file, 'a') as fp:
        group = tmplt.format(mode)
        if group not in fp:
            fp.create_dataset(group, norm.shape, dtype=norm.dtype,
                              compression="gzip")
        fp[group][:] = norm
        if m != 0:
            group = nmtmplt.format(mode)
            if group not in fp:
                fp.create_dataset(group, norm.shape, dtype=norm.dtype,
                                  compression="gzip")
            fp[group][:] = nmnorm
    print("")
print("done")
