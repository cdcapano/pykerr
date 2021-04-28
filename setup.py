import setuptools

# get version
with open("pykerr/_version.py", "r") as fh:
    vstr = fh.read().strip()
    vstr = vstr.split('=')[1].strip()
    version = vstr.replace("'", "").replace('"', "")

setuptools.setup(
    name="pykerr",
    version=version,
    url="https://github.com/cdcapano/pykerr.git",
    author="Collin Capano",
    author_email="collin.capano@aei.mpg.de",
    description="QNM frequencies and spheroidal harmonics of Kerr black holes.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'h5py'],
    include_package_data=True,
    package_data={'': ['data/*.hdf']},
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
    ],
)
