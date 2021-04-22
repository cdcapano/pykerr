#! /usr/bin/env python

import argparse
import numpy
import h5py
import glob
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-directory', required=True,
                     help='The directory to read. Will load all dat files '
                          'in the given directory.')
parser.add_argument('-o', '--output-file', required=True,
                    help='The file to write out to. Must be an hdf file.')


opts = parser.parse_args()

datfiles = glob.glob('{}/*.dat'.format(opts.input_directory))

# regex for extracting n, l, m from file name
regex = re.compile(r'n([1-9])l([2-9])m(m*[0-9])')

# according to https://centra.tecnico.ulisboa.pt/network/grit/files/ringdown/,
# the file format for the Kerr QNM are:
# a/M, Re[M omega}, Im[M omega], Re[Alm], Im[Alm] 
dtype = [('spin', numpy.float32),
         ('omegaR', numpy.float32), ('omegaI', numpy.float32),
         ('ReAlm', numpy.float32), ('ImAlm', numpy.float32)]

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
for lmn, data in qnm.items():
    # spin
    tmplt = lmn + '/{}'
    group = tmplt.format('spin')
    out.create_dataset(group, data.shape, dtype=numpy.float32,
                       compression="gzip")
    out[group][:] = qnm[lmn]['spin']
    # omega
    group = tmplt.format('omega')
    out.create_dataset(group, data.shape, dtype=numpy.complex64,
                       compression="gzip")
    out[group][:] = qnm[lmn]['omegaR'] + 1j*qnm[lmn]['omegaI']
    # angular separtion constant
    group = tmplt.format('alm')
    out.create_dataset(group, data.shape, dtype=numpy.complex64,
                       compression='gzip')
    out[group][:] = qnm[lmn]['ReAlm'] + 1j*qnm[lmn]['ImAlm']
out.close()
