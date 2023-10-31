# pykerr

[![DOI](https://zenodo.org/badge/359106882.svg)](https://zenodo.org/doi/10.5281/zenodo.10056494)

This package provides functions to get the frequency, damping time, and spheroidal harmonics of Kerr black holes. Solutions for the `l=2` to `l=7` modes are provided, including all `m=[-l, ..., l]` and up to 7 overtones (where `n=0` is the fundamental mode), for dimensionless black hole spins up to +/- 0.9997. Currently, only spin weights of `s=-2` are supported.

## Installation

The easiest way to install pykerr is via pip:
```
pip install pykerr
```
or conda:
```
conda install -c conda-forge pykerr
```

You can also install from source by cloning the repository at https://github.com/cdcapano/pykerr. Required packages are listed in the `requirements.txt` file, which you can install by running `pip install -r requirements.txt` from within the source code directory, followed by `pip install .`.

## Examples

 1. Get the frequency and damping time of a Kerr black hole:

```
>>> import pykerr
>> pykerr.qnmfreq(200., 0.7, 2, 2, 0)
86.04823229677822
>>> pykerr.qnmtau(200., 0.7, 2, 2, 0)
0.012192884850631896
```

 2. Get the spheroidal harmonics for the same black hole, viewed from a polar angle of pi/3 and an azimuthal angle of 0:

```
>>> pykerr.spheroidal(numpy.pi/3, 0.7, 2, 2, 0, phi=0.)
(0.3378406925922286-0.0007291958007333236j)
```

## Details

`pykerr` uses pre-tabulated values for the Kerr QNM frequencies and angular separation constants. These are used to obtain solutions for the spheroidal harmonics using the recursion relation given by Eq. 18 in [Leaver (1985)](https://doi.org/10.1098/rspa.1985.0119) [1]. The pre-tabulated values for the QNM frequency, damping time, and angular separation constant comes from [Berti et al. (2006)](https://doi.org/10.1103/PhysRevD.73.064030) [2], made available as text files on the [GRIT Ringdown website](https://centra.tecnico.ulisboa.pt/network/grit/files/ringdown/). Those files are repackaged into compressed hdf files that are released with this package. A cubic spline is applied to the pretabulated values to provide fast evaluation of the spheroidal harmonics and QNM frequencies at any arbitrary spin `<= 0.9997`. Pre-tabulated normalization constants for the `s=-2` spheroidal harmonics are also provided, with a cubic spline being used to interpolate them.

`pykerr` does not calculate QNM frequencies and angular separation constants. For that, see the various Mathematica packages that are publicly available or the [qnm](https://pypi.org/project/qnm/) package, which can be installed via pip. Interpolated values have been checked against [London (2017)](https://github.com/llondon6/kerr_public).

## Conventions

This uses the convention that the -m and +m modes are related to each other by:
```
f_{l-mn} = -f_{lmn}
tau_{l-mn} = tau_{lmn}
A_{l-mn} = A*_{lmn}
```
with dimensionless spin a/M being positive or negative. Negative spin means the perturbation is counter-rotating with respect to the black hole, while positive spin means the perturbation is co-rotating.

## Custom tabulation

If you would like `pykerr` to use your own tabulated values, clone the code from soucre, then add or replace the hdf files stored in `pykerr/data`. The files should be named `l{l}.hdf`, where `{l}` is the l index. The hdf files need to have a top-level `spin` dataset with spins stored as 10^4 times the spin value. Each mode should be provided as a separate group, named `{l}{m}{n}`. In that group, there should be complex datasets called `omega` and `alm` that provides the dimensionless complex angular frequencies and angular separation constants, respectively. See the `convert_to_hdf.py` in the `scripts` directory, which is used to convert the text files from the GRIT website into the correct format, as an example. Once your files are implace, install pykerr using `pip install .`. That will copy your files to the install directory to be used by your code at runtime.

If you add values for spins beyond 0.9997 and would like pykerr to support them, change `pykerr.qnm.MAX_SPIN` appropriately. This can also be done at run time. It is recommended that at least three data points be provided beyond your spin limit to avoid boundary effects from the cubic spline.

To add pre-tabulated normalization constants to the data files use the `tabulate_norms.py` script, provided in the `scripts` directory.

## Attribution

If you use `pykerr` in your work, please cite DOI 10.5281/zenodo.10056494 for the latest version, or the DOI specific to the release you used. Authorship, citation format, and DOI for all versions are available at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10056494).


## References

1. E. W. Leaver, "An Analytic representation for the quasi normal modes of Kerr black holes", [Proc. R. Soc. Lond. A 402 285-298](https://doi.org/10.1098/rspa.1985.0119) (1985).

2. E. Berti, V. Cardoso, and C. M. Will, "On gravitational-wave spectroscopy of massive black holes with the space interferometer LISA", [PRD 73 064030](https://doi.org/10.1103/PhysRevD.73.064030) (2006), [arXiv:0512160](https://arxiv.org/abs/gr-qc/0512160) [gr-qc].
