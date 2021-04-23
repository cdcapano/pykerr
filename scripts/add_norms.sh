#! /usr/bin/env bash
set -e
for ll in 2 3 4 5 6 7; do
    mode=l${ll}
    nohup python tabulate_norms.py -i ../pykerr/data/${mode}.hdf  -s 2 &> tabnorm-${mode}.out &
done
