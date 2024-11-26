## Kitsune-specific tests ##

The tests here are end-to-end tests for the Kitsune-specific extensions and 
code generation. They are intended to be used to test both correctness and 
performance regressions. 

The tests here include a mix of C, C++, Fortran, Cuda, Hip and Kokkos. The Cuda
(.cu) and Hip (.hip) tests are only intended for performance comparisons and 
have nothing to do with Kitsune (in fact, if enabled, they are compiled with
`nvcc` and `hipcc` respectively). 

## Building ##

TODO: Describe how to build the tests. 

## Organization ###

The subdirectories in this test contain equivalent implementations of the same 
test in a number of different languages. Tests which do not contain equivalent
implementations in different languages may be placed directly in this 
directory. 

### Kitsune tests ###

The names of test files containing Kitsune-specific extensions must end with 
`.kit.cpp` (for C++) and `.kit.c` (for C). This is simply to distinguish them
from any standard C++ (or C) files that  may be present in the directory. Only
the files with this extension will be compiled with all the tapir targets being
tested.

### Kokkos tests ###

The names of test files containing Kokkos must end with `.kokkos.cpp`. Kitsune's
`-fkokkos-mode` treats Kokkos as its own language, but the normal file extension
cannot be used to distinguish such files from normal C++. 

[_TODO: Figure out what do with tests that contain Kokkos views_]
