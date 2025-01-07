# Kitsune-specific tests #

This directory contains end-to-end tests for Kitsune. They are intended to test
for both correctness and (some) performance regressions. The section for 
[developers](#Developer Guide) contains more information about the test suite
itself, how it is organized and the contents. The sections on 
[building](#Building) and [running](#Running) the test suite should be 
sufficient to just use the test suite.

## Building ##

The most straightforward way of building the LLVM test suite to run _only_ the 
Kitsune-specific tests would be something like this:

```
$ cmake -DCMAKE_C_COMPILER=/path/to/kitcc \
    -DCMAKE_CXX_COMPILER=/path/to/kit++ \
    -DTEST_SUITE_SUBDIRS=Kitsune \
    /path/to/kitsune-test-suite
```

This will test all the tapir targets that kitsune has been built with. This may
may not always be desirable. For instance, if both the `cuda` and `hip` backends
have been built, this will attempt to run tests on both NVIDIA and AMD GPU's. In
most cases, only one of these will be present on the test machine. Also, this 
will test the `serial` tapir target (which is always built) which, for some 
tests, can be very slow. There are a number of ways to test only a subset of the
tapir targets. 

`-DKITSUNE_SKIP_TAPIR_TARGETS=<targets>` can be used to test all tapir targets
that have been built _except_ those in `<targets>`. `<targets>` is a 
semicolon-separated list.

`-DKITSUNE_TEST_TAPIR_TARGETS=<targets>` can be used to test _only_ those 
tapir targets that have been built _and_ are present in `<targets>`. 
`<targets>` is a semicolon-separated list. 

In general, only one of `-DKITSUNE_SKIP_TAPIR_TARGETS` and 
`-DKITSUNE_TEST_TAPIR_TARGETS` should be specified. Providing both is allowed, 
but it is unlikely that any good will come of this.

By default, all the frontends that Kitsune was built with are tested. Currently,
frontends for C (`kitcc`) and C++ (`kit++`) are available. A Fortran frontend
(`kitfc`) is under development and should also be available in the future. The
test suite can be restricted to testing only some of the built frontends.

`-DKITSUNE_SKIP_FRONTENDS=<frontends>` can be used to test all frontends that 
have been built _except_ those in `<frontends>`. `<frontends>` is a c

`-DKITSUNE_TEST_FRONTENDS=<frontends>` can be used to test _only_ those
frontends that have been built _and_ are present in `<frontends>`. `<frontends>`
is a semciolon-separated list.

`-DKITSUNE_TEST_INPUT_SIZE` can be used to change the input sizes to some of the 
tests, particularly those in `Benchmarks/`. The accepted values are `"default"`
and `"small"`. The default sizes are useful for performance comparisons, but if 
one is only interested in checking for correctness, the `small` input size may
be the better choice since the test suite will run much faster in such cases.

`-DKITSUNE_TEST_KOKKOS_MODE` can be used to enable or disable Kitsune's Kokkos
mode where it recognizes some Kokkos constructs and generates custom code for 
them. By default, this is set to `ON` and will test Kokkos mode if it has been
enabled in Kitsune. When set to `OFF`, Kokkos mode will not be tested.

`-DKITSUNE_BENCHMARK=ON` is required to enable the "benchmarking mode" of the 
test suite. The `Benchmarks/` directory contains tests intended for performance
comparisons with, for instance, equivalent cuda (`.cu`) implementations. This 
typically requires `nvcc` or `hipcc` to be available on the system. Turning this
on will also run the Kokkos tests in vanilla mode (in addition to Kitsune's
`-fkokkos-mode`). This will use the Kokkos backend that Kitsune was built with. 
This is typically one with GPU support. _[[TODO: Explain the Kokkos details \
better]]_

## Running ##

Running the tests in the test suite requires LLVM's `lit` utility. This can be
found in the `bin/` directory within Kitsune's build directory, 
`/path/to/kitsune/build/bin/llvm-lit`. This utility is not installed, even if 
`-DLLVM_INSTALL_UTILS=ON` was set when building Kitsune. Another option is to
install `lit`[^1]. Some Linux distributions such as Arch and Debian package this
together with the LLVM package. Otherwise it can be obtained from PyPi, for 
example, using `pip`

```
$ pip install --user lit
```

Once installed, please ensure that `lit` is in `$PATH`. A script to run the
Kitsune tests is available in the Kitsune subdirectory of the _build_
directory. In addition to running the tests, it also runs a post-processing
script that collects useful statistics into a report file in
`Kitsune/report.json`. It can be run from the build directory as shown

```
$ ./Kitsune/run-tests
```

The `run-tests` script will produce a report, `Kitsune/report.json`, containing
statistics collected at both compile and run-time. The report collects data from
both the raw report produced by the test suite in `Kitsune/report-ts.json` and 
the output produced by the tests themselves. Alternatively, the tests can
also be run manually by calling `lit` directly (note that this will not produce
any report).

```
$ lit /path/to/kitsune-test-suite/build/Kitsune
```

`lit` supports some other options that may be useful. Use `lit --help` to see
a full list. 


[^1]: LLVM's lit utility is written in Python, and as such, is somewhat 
independent of LLVM itself. As long as one doesn't obtain a very old version of
lit, it should work even if installed separately.

## Developer Guide ##

This section contains notes on the organization of the Kitsune-specific tests
and information on how to add new tests.

## Organization ##

The subdirectories in this test contain equivalent implementations of the same 
test in a number of different languages. Tests which do not contain equivalent
implementations in different languages may be placed directly in this 
directory. 

## Kitsune tests ##

The names of test files containing Kitsune-specific extensions must end with 
`.kit.c`, `.kit.cpp`, or `.kit.f90` for C, C++ and Fortran respectively. This 
is simply to distinguish them from any standard C++ (or C) files that may be
present in the directory. Only the files with this extension will be compiled
``with all the tapir targets being tested.

## Kokkos tests ##

The names of test files containing Kokkos must end with `.kokkos.cpp`. Kitsune's
`-fkokkos-mode` treats Kokkos as its own language, but the normal file extension
cannot be used to distinguish such files from normal C++. 

[_TODO: Figure out what do with tests that contain Kokkos views_]
