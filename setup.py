import setuptools

setuptools.setup(
    name="pykerr",
    version="0.1.0",
    url="",
    author="Collin Capano",
    author_email="collin.capano@aei.mpg.de",
    description="QNM utilties for gravitational-wave astronomy.",
    #long_description=open('DESCRIPTION.rst').read(),
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'h5py'],
    include_package_data=True,
    package_data={'': ['data/*.hdf']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
    ],
)
