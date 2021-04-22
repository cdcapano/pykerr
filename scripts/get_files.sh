#! /usr/bin/env bash
set -e
for ll in 2 3 4 5 6 7; do
    mode=l${ll}
    mkdir -p ${mode}
    wget http://blackholes.ist.utl.pt/Webpagecodes/${mode}.tar.gz
    tar -xzvf ${mode}.tar.gz -C ${mode}
    python convert_to_hdf.py -i ${mode} -o ../pykerr/data/${mode}.hdf
done
