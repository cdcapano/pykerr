# pyqnm
Utilities for quasi normal modes of Kerr black holes.

## Installation

First make sure you have the needed packages installed by running:

```
pip install -r requirements.txt
```

Next, run the `scripts/get_files.sh` to download the QNM data from the [GRIT Ringdown website](https://centra.tecnico.ulisboa.pt/network/grit/files/ringdown/):

```
cd scripts
bash get_files.sh
cd ..
```

Now install the code by running

```
python setup.py install
```

## Example

```
>>> import pyqnm
>> pyqnm.kerr_freq(200., 0.7, 2, 2, 0)
86.04823229677822
>>> pyqnm.kerr_tau(200., 0.7, 2, 2, 0)
0.012192884850631896
```

## Conventions

This uses the convention that the -m and +m modes are related to each other by:
```
f_{l-mn} = -f_{lmn}
tau_{l-mn} = tau_{lmn}
```
with dimensionless spin a/M being positive or negative. Negative spin means the perturbation is counterrotating with respect to the black hole, while positive spin means the perturbation is corotating.
