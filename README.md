# lsCOMP

<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>

This branch contains the compiled shared object (or dynamic link library) for lsCOMP compressor, for RHEL 9 OS, CUDA 12.1.1, and GCC 11.5.0 (within Red Hat).

To use this pre-compiled shared object, you can replace the path of ```.so``` in line 37 of ```lsCOMP/python/lsCOMP.py``` in the main branch to the compiled ```liblsCOMP-cuda12.9-rhel9.3-gcc11.5.so``` in this branch.

And the lsCOMP can be launched via the provided Python script.