#! /usr/bin/env python

import argparse
import numpy
import h5py
import glob
import re
import os

"""Converts text files downloaded from Cardoso's webpage into hdf file."""

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-directory', required=True,
                     help='The directory to read. Will load all dat files '
                          'in the given directory.')
parser.add_argument('-o', '--output-file', required=True,
                    help='The file to write out to. Must be an hdf file.')
parser.add_argument('--thin', type=int, default=None,
                    help="Thin the tabulated values by the given amount. "
                         "Must be an odd integer > 1, so as to preserve "
                         "end points and zero spin.")
opts = parser.parse_args()

if opts.thin is not None:
    if opts.thin < 0 or (-1)**opts.thin > 0:
        raise ValueError('--thin must be a positive odd integer')

datfiles = glob.glob('{}/*.dat'.format(opts.input_directory))

# regex for extracting n, l, m from file name
regex = re.compile(r'n([1-9])l([2-9])m(m*[0-9])')

# according to https://centra.tecnico.ulisboa.pt/network/grit/files/ringdown/,
# the file format for the Kerr QNM are:
# a/M, Re[M omega}, Im[M omega], Re[Alm], Im[Alm] 
dtype = [('spin', float), ('omegaR', float), ('omegaI', float),
         ('ReAlm', float), ('ImAlm', float)]

qnm = {}
for datfn in datfiles:
    # extract the lmn from the file name
    match = regex.match(os.path.basename(datfn))
    if match is None:
        raise ValueError("lmn could not be parsed from file name {} "
                         .format(datfn))
    # subtract 1 from the overtone number because Cardosa counts overtones
    # starting from 1
    nn = int(match.group(1)) - 1
    ll = int(match.group(2))
    mm = int(match.group(3).replace('m', '-'))
    data = numpy.loadtxt(datfn, dtype=dtype)
    # some values that should be 0 are set to a tiny amount; correct them
    # to be zero
    for name, _ in dtype:
        mask = abs(data[name]) < 1e-12
        data[name][mask] = 0.
    # we'll use the convention that the -m modes correspond to negative spin
    if mm < 0:
        data['spin'] *= -1
        # flip and exclude the 0 frequency, since it should be in the +m
        data = data[::-1][:-1]
    lmn = '{}{}{}'.format(ll, abs(mm), nn)
    if lmn in qnm:
        # already there, concatenate
        if mm < 0:
            qnm[lmn] = numpy.concatenate((data, qnm[lmn]))
        else:
            qnm[lmn] = numpy.concatenate((qnm[lmn], data))
    else:
        qnm[lmn] = data

# now write
out = h5py.File(opts.output_file, 'w')
group = '{}/{}'
# we'll store the spins separately since they're the same for all m != 0
# and m = 0 modes
sp = None
spm0 = None
for lmn, data in qnm.items():
    print(lmn)
    tmplt = lmn + '/{}'
    if opts.thin is not None:
        # perserve the last 9 values close to the boundaries, then increase
        # the thinning to the desired amount
        dk = 9
        keep = []
        ti = 0
        while ti < opts.thin:
            keep.append(data[ti*dk:(ti+1)*dk:ti+1])
            ti += 1
        keep.append(data[ti*dk:-ti*dk:opts.thin])
        while ti > 1:
            keep.append(data[-ti*dk:-(ti-1)*dk:ti])
            ti -= 1
        keep.append(data[-dk:])
        data = numpy.concatenate(keep)
    # spin
    thissp = numpy.round(10000*data['spin']).astype(int)
    if lmn[1] == '0':
        if spm0 is None:
            spm0 = thissp
        elif not (spm0 == thissp).all():
            raise ValueError('got different spins for lmn {}'.format(lmn))
    else:
        if sp is None:
            sp = thissp
        elif not (sp == thissp).all():
            raise ValueError('got different spins for lmn {}'.format(lmn))
    # omega
    group = tmplt.format('omega')
    out.create_dataset(group, data.shape, dtype=numpy.complex128,
                       compression="gzip")
    out[group][:] = data['omegaR'] + 1j*data['omegaI']
    # angular separtion constant
    group = tmplt.format('alm')
    out.create_dataset(group, data.shape, dtype=numpy.complex128,
                       compression='gzip')
    out[group][:] = data['ReAlm'] + 1j*data['ImAlm']
# write the spins
out.create_dataset('spin', sp.shape, dtype=numpy.int16, compression="gzip")
out['spin'][:] = sp
out.create_dataset('spinm0', spm0.shape, dtype=numpy.int16, compression="gzip")
out['spinm0'][:] = spm0
out.close()
